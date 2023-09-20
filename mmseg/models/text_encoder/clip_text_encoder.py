# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from torch.nn import functional as F

from mmseg.registry import MODELS
from mmseg.utils import get_classes, get_predefined_templates, tokenizer


@MODELS.register_module()
class CLIPTextEncoder(BaseModule):
    """A text encoder with transformer architecture to encode the label text.

    Modified from https://github.com/MendelXu/SAN/blob/main/san/model/clip_utils/classifier.py # noqa:E501
    Copyright (c) 2023 MendelXu.
    Licensed under the MIT License

    Args:
        dataset_name: (str|None): The name of the dataset to which
            the data belongs.
        vocabulary: (List[str]|None): The list of class names. Default: None.
        templates: (List[str]|None): The prompt template used for labels.
            Default: None.
        total_vocab_size: (int): Number of all words used by the pre-trained
            model. Default: 49408 (CLIP).
        context_length: (int): The max length of prompt text.
            Default: 77 (CLIP).
        embed_dims: (int): Width of transformer model. Default: 512.
        num_layers: (int): Depth of transformer. Default: 12,
        num_heads: (int): Number of attention heads in transformer.
            Default: 8,
        mlp_ratio: (int) Ratio of mlp hidden dim to embedding dim in
            transformer. Default: 4,
        output_dims: (int) Dim of output text embeddings. Default: 512,
        cache_feature: (bool) Whether to save class embeddings in cache.
            Default: True,
        cat_bg: (bool) Whether to add background embedding. Default: True.
        norm_cfg (dict|None): Config for norm layer. Default: dict(type='LN')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 dataset_name: str = None,
                 vocabulary: List[str] = None,
                 templates: str = 'vild',
                 total_vocab_size: int = 49408,
                 context_length: int = 77,
                 embed_dims: int = 512,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 output_dims: int = 512,
                 cache_feature: bool = True,
                 cat_bg: bool = True,
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: dict = None):
        super().__init__(init_cfg)
        if isinstance(templates, List):
            self.templates = templates
        else:
            self.templates = get_predefined_templates(templates)

        assert dataset_name is not None or vocabulary is not None, \
            "text_encoder required either 'dataset_name' or 'vocabulary'"
        assert dataset_name is None or vocabulary is None, \
            "there is conflict between 'dataset_name' and 'vocabulary'"
        self.dataset_name = dataset_name
        self.vocabulary = vocabulary
        self.num_pos = context_length
        self.token_embedding = nn.Embedding(total_vocab_size, embed_dims)
        self.positional_embedding = nn.Parameter(
            torch.empty(context_length, embed_dims))
        self.text_projection = nn.Parameter(
            torch.empty(embed_dims, output_dims))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer = ModuleList()
        self.register_buffer(
            'attn_mask', self.build_attention_mask(), persistent=False)
        for i in range(num_layers):
            self.transformer.append(
                BaseTransformerLayer(
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        batch_first=False,
                        bias=True),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=mlp_ratio * embed_dims,
                        act_cfg=dict(type='QuickGELU')),
                    operation_order=('norm', 'self_attn', 'norm', 'ffn')))
        self.ln_final = build_norm_layer(
            norm_cfg, embed_dims, postfix='_final')[1]

        self.cache_feature = cache_feature
        if self.cache_feature:
            self.cache = {}

        self._freeze()

        self.cat_bg = cat_bg
        if self.cat_bg:
            self.bg_embed = nn.Parameter(
                torch.randn(1, self.text_projection.shape[1]))

    @property
    def ln_final(self):
        return getattr(self, self.final_name)

    def build_attention_mask(self):
        """lazily create causal attention mask, with full attention between the
        tokens.

        pytorch uses additive attention mask; fill with -inf
        """
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def init_weights(self):
        if self.cat_bg:
            nn.init.normal_(
                self.bg_embed,
                std=self.bg_embed.shape[1]**-0.5,
            )
        if isinstance(self.init_cfg, dict) and \
                self.init_cfg.get('type') == 'Pretrained_Part':
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            state_dict = checkpoint.copy()
            para_prefix = 'text_encoder'
            prefix_len = len(para_prefix) + 1
            for k, v in checkpoint.items():
                state_dict.pop(k)
                if para_prefix in k:
                    state_dict[k[prefix_len:]] = v

            load_state_dict(self, state_dict, strict=False, logger=None)

        else:
            super().init_weights()

    @torch.no_grad()
    def encode_text(self, text, normalize=False):
        """encode class token."""

        embed_device = self.token_embedding.weight.device
        x = self.token_embedding(
            text.to(embed_device))  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        for block in self.transformer:
            x = block(query=x, attn_masks=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding
        # (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def template_encode(self, vocabulary):
        """Prompt engineering."""
        text_embed_bucket = []
        for template in self.templates:
            text_inputs = tokenizer.tokenize(
                [template.format(noun) for noun in vocabulary])
            text_embed = self.encode_text(text_inputs, normalize=True)
            text_embed_bucket.append(text_embed)
        text_embed = torch.stack(text_embed_bucket).mean(dim=0)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        return text_embed

    def forward(self):
        """Forward function."""
        if self.dataset_name is None:  # encoding vocabulary directly
            class_names = self.vocabulary
            if self.cache_feature:
                new_classes = [
                    word for word in class_names if word not in self.cache
                ]
                if len(new_classes) > 0:
                    class_embeds = self.template_encode(new_classes)
                    self.cache.update(dict(zip(new_classes, class_embeds)))
                class_embeds = torch.stack(
                    [self.cache[word] for word in class_names])
            else:
                class_embeds = self.template_encode(class_names)

        else:  # encoding the classes of the dataset
            class_names = get_classes(self.dataset_name)
            if class_names[0] == 'background':
                class_names = class_names[1:]
            if self.cache_feature:
                if self.dataset_name not in self.cache:
                    class_embeds = self.template_encode(class_names)
                    self.cache[self.dataset_name] = class_embeds
                else:
                    class_embeds = self.cache[self.dataset_name]
            else:
                class_embeds = self.template_encode(class_names)

        if self.cat_bg:
            class_embeds = torch.cat([class_embeds, self.bg_embed])
            class_embeds = F.normalize(class_embeds, p=2, dim=-1)
        return self.logit_scale.exp() * class_embeds


@MODELS.register_module()
class QuickGELU(nn.Module):
    # From https://github.com/openai/CLIP/blob/main/clip/model.py
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
