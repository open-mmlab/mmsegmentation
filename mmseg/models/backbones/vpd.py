# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------
# Adapted from https://github.com/wl-zhao/VPD/blob/main/vpd/models.py
# Original licence: MIT License
# ------------------------------------------------------------------------------

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader, load_checkpoint

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType

try:
    from ldm.modules.diffusionmodules.util import timestep_embedding
    from ldm.util import instantiate_from_config
    has_ldm = True
except ImportError:
    has_ldm = False


def register_attention_control(model, controller):
    """Registers a control function to manage attention within a model.

    Args:
        model: The model to which attention is to be registered.
        controller: The control function responsible for managing attention.
    """

    def ca_forward(self, place_in_unet):
        """Custom forward method for attention.

        Args:
            self: Reference to the current object.
            place_in_unet: The location in UNet (down/mid/up).

        Returns:
            The modified forward method.
        """

        def forward(x, context=None, mask=None):
            h = self.heads
            is_cross = context is not None
            context = context or x  # if context is None, use x

            q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
            q, k, v = (
                tensor.view(tensor.shape[0] * h, tensor.shape[1],
                            tensor.shape[2] // h) for tensor in [q, k, v])

            sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if mask is not None:
                mask = mask.flatten(1).unsqueeze(1).repeat(h, 1, 1)
                max_neg_value = -torch.finfo(sim.dtype).max
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            attn_mean = attn.view(h, attn.shape[0] // h,
                                  *attn.shape[1:]).mean(0)
            controller(attn_mean, is_cross, place_in_unet)

            out = torch.matmul(attn, v)
            out = out.view(out.shape[0] // h, out.shape[1], out.shape[2] * h)
            return self.to_out(out)

        return forward

    def register_recr(net_, count, place_in_unet):
        """Recursive function to register the custom forward method to all
        CrossAttention layers.

        Args:
            net_: The network layer currently being processed.
            count: The current count of layers processed.
            place_in_unet: The location in UNet (down/mid/up).

        Returns:
            The updated count of layers processed.
        """
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        if hasattr(net_, 'children'):
            return sum(
                register_recr(child, 0, place_in_unet)
                for child in net_.children())
        return count

    cross_att_count = sum(
        register_recr(net[1], 0, place) for net, place in [
            (child, 'down') if 'input_blocks' in name else (
                child, 'up') if 'output_blocks' in name else
            (child,
             'mid') if 'middle_block' in name else (None, None)  # Default case
            for name, child in model.diffusion_model.named_children()
        ] if net is not None)

    controller.num_att_layers = cross_att_count


class AttentionStore:
    """A class for storing attention information in the UNet model.

    Attributes:
        base_size (int): Base size for storing attention information.
        max_size (int): Maximum size for storing attention information.
    """

    def __init__(self, base_size=64, max_size=None):
        """Initialize AttentionStore with default or custom sizes."""
        self.reset()
        self.base_size = base_size
        self.max_size = max_size or (base_size // 2)
        self.num_att_layers = -1

    @staticmethod
    def get_empty_store():
        """Returns an empty store for holding attention values."""
        return {
            key: []
            for key in [
                'down_cross', 'mid_cross', 'up_cross', 'down_self', 'mid_self',
                'up_self'
            ]
        }

    def reset(self):
        """Resets the step and attention stores to their initial states."""
        self.cur_step = 0
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """Processes a single forward step, storing the attention.

        Args:
            attn: The attention tensor.
            is_cross (bool): Whether it's cross attention.
            place_in_unet (str): The location in UNet (down/mid/up).

        Returns:
            The unmodified attention tensor.
        """
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= (self.max_size)**2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        """Processes and stores attention information between steps."""
        if not self.attention_store:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                self.attention_store[key] = [
                    stored + step for stored, step in zip(
                        self.attention_store[key], self.step_store[key])
                ]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        """Calculates and returns the average attention across all steps."""
        return {
            key: [item for item in self.step_store[key]]
            for key in self.step_store
        }

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        """Allows the class instance to be callable."""
        return self.forward(attn, is_cross, place_in_unet)

    @property
    def num_uncond_att_layers(self):
        """Returns the number of unconditional attention layers (default is
        0)."""
        return 0

    def step_callback(self, x_t):
        """A placeholder for a step callback.

        Returns the input unchanged.
        """
        return x_t


class UNetWrapper(nn.Module):
    """A wrapper for UNet with optional attention mechanisms.

    Args:
        unet (nn.Module): The UNet model to wrap
        use_attn (bool): Whether to use attention. Defaults to True
        base_size (int): Base size for the attention store. Defaults to 512
        max_attn_size (int, optional): Maximum size for the attention store.
            Defaults to None
        attn_selector (str): The types of attention to use.
            Defaults to 'up_cross+down_cross'
    """

    def __init__(self,
                 unet,
                 use_attn=True,
                 base_size=512,
                 max_attn_size=None,
                 attn_selector='up_cross+down_cross'):
        super().__init__()

        assert has_ldm, 'To use UNetWrapper, please install required ' \
            'packages via `pip install -r requirements/optional.txt`.'

        self.unet = unet
        self.attention_store = AttentionStore(
            base_size=base_size // 8, max_size=max_attn_size)
        self.attn_selector = attn_selector.split('+')
        self.use_attn = use_attn
        self.init_sizes(base_size)
        if self.use_attn:
            register_attention_control(unet, self.attention_store)

    def init_sizes(self, base_size):
        """Initialize sizes based on the base size."""
        self.size16 = base_size // 32
        self.size32 = base_size // 16
        self.size64 = base_size // 8

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """Forward pass through the model."""
        diffusion_model = self.unet.diffusion_model
        if self.use_attn:
            self.attention_store.reset()
        hs, emb, out_list = self._unet_forward(x, timesteps, context, y,
                                               diffusion_model)
        if self.use_attn:
            self._append_attn_to_output(out_list)
        return out_list[::-1]

    def _unet_forward(self, x, timesteps, context, y, diffusion_model):
        hs = []
        t_emb = timestep_embedding(
            timesteps, diffusion_model.model_channels, repeat_only=False)
        emb = diffusion_model.time_embed(t_emb)
        h = x.type(diffusion_model.dtype)
        for module in diffusion_model.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = diffusion_model.middle_block(h, emb, context)
        out_list = []
        for i_out, module in enumerate(diffusion_model.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if i_out in [1, 4, 7]:
                out_list.append(h)
        h = h.type(x.dtype)
        out_list.append(h)
        return hs, emb, out_list

    def _append_attn_to_output(self, out_list):
        avg_attn = self.attention_store.get_average_attention()
        attns = {self.size16: [], self.size32: [], self.size64: []}
        for k in self.attn_selector:
            for up_attn in avg_attn[k]:
                size = int(math.sqrt(up_attn.shape[1]))
                up_attn = up_attn.transpose(-1, -2).reshape(
                    *up_attn.shape[:2], size, -1)
                attns[size].append(up_attn)
        attn16 = torch.stack(attns[self.size16]).mean(0)
        attn32 = torch.stack(attns[self.size32]).mean(0)
        attn64 = torch.stack(attns[self.size64]).mean(0) if len(
            attns[self.size64]) > 0 else None
        out_list[1] = torch.cat([out_list[1], attn16], dim=1)
        out_list[2] = torch.cat([out_list[2], attn32], dim=1)
        if attn64 is not None:
            out_list[3] = torch.cat([out_list[3], attn64], dim=1)


class TextAdapter(nn.Module):
    """A PyTorch Module that serves as a text adapter.

    This module takes text embeddings and adjusts them based on a scaling
    factor gamma.
    """

    def __init__(self, text_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim, text_dim), nn.GELU(),
            nn.Linear(text_dim, text_dim))

    def forward(self, texts, gamma):
        texts_after = self.fc(texts)
        texts = texts + gamma * texts_after
        return texts


@MODELS.register_module()
class VPD(BaseModule):
    """VPD (Visual Perception Diffusion) model.

    .. _`VPD`: https://arxiv.org/abs/2303.02153

    Args:
        diffusion_cfg (dict): Configuration for diffusion model.
        class_embed_path (str): Path for class embeddings.
        unet_cfg (dict, optional): Configuration for U-Net.
        gamma (float, optional): Gamma for text adaptation. Defaults to 1e-4.
        class_embed_select (bool, optional): If True, enables class embedding
            selection. Defaults to False.
        pad_shape (Optional[Union[int, List[int]]], optional): Padding shape.
            Defaults to None.
        pad_val (Union[int, List[int]], optional): Padding value.
            Defaults to 0.
        init_cfg (dict, optional): Configuration for network initialization.
    """

    def __init__(self,
                 diffusion_cfg: ConfigType,
                 class_embed_path: str,
                 unet_cfg: OptConfigType = dict(),
                 gamma: float = 1e-4,
                 class_embed_select=False,
                 pad_shape: Optional[Union[int, List[int]]] = None,
                 pad_val: Union[int, List[int]] = 0,
                 init_cfg: OptConfigType = None):

        super().__init__(init_cfg=init_cfg)

        assert has_ldm, 'To use VPD model, please install required packages' \
            ' via `pip install -r requirements/optional.txt`.'

        if pad_shape is not None:
            if not isinstance(pad_shape, (list, tuple)):
                pad_shape = (pad_shape, pad_shape)

        self.pad_shape = pad_shape
        self.pad_val = pad_val

        # diffusion model
        diffusion_checkpoint = diffusion_cfg.pop('checkpoint', None)
        sd_model = instantiate_from_config(diffusion_cfg)
        if diffusion_checkpoint is not None:
            load_checkpoint(sd_model, diffusion_checkpoint, strict=False)

        self.encoder_vq = sd_model.first_stage_model
        self.unet = UNetWrapper(sd_model.model, **unet_cfg)

        # class embeddings & text adapter
        class_embeddings = CheckpointLoader.load_checkpoint(class_embed_path)
        text_dim = class_embeddings.size(-1)
        self.text_adapter = TextAdapter(text_dim=text_dim)
        self.class_embed_select = class_embed_select
        if class_embed_select:
            class_embeddings = torch.cat(
                (class_embeddings, class_embeddings.mean(dim=0,
                                                         keepdims=True)),
                dim=0)
        self.register_buffer('class_embeddings', class_embeddings)
        self.gamma = nn.Parameter(torch.ones(text_dim) * gamma)

    def forward(self, x):
        """Extract features from images."""

        # calculate cross-attn map
        if self.class_embed_select:
            if isinstance(x, (tuple, list)):
                x, class_ids = x[:2]
                class_ids = class_ids.tolist()
            else:
                class_ids = [-1] * x.size(0)
            class_embeddings = self.class_embeddings[class_ids]
            c_crossattn = self.text_adapter(class_embeddings, self.gamma)
            c_crossattn = c_crossattn.unsqueeze(1)
        else:
            class_embeddings = self.class_embeddings
            c_crossattn = self.text_adapter(class_embeddings, self.gamma)
            c_crossattn = c_crossattn.unsqueeze(0).repeat(x.size(0), 1, 1)

        # pad to required input shape for pretrained diffusion model
        if self.pad_shape is not None:
            pad_width = max(0, self.pad_shape[1] - x.shape[-1])
            pad_height = max(0, self.pad_shape[0] - x.shape[-2])
            x = F.pad(x, (0, pad_width, 0, pad_height), value=self.pad_val)

        # forward the denoising model
        with torch.no_grad():
            latents = self.encoder_vq.encode(x).mode().detach()
        t = torch.ones((x.shape[0], ), device=x.device).long()
        outs = self.unet(latents, t, context=c_crossattn)

        return outs
