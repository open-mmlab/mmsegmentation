"""
Multi-Modal Swin Transformer with MoE - MMSeg 1.x Version

Migrated from mmseg 0.x to 1.x:
- Registry: BACKBONES -> MODELS
- BaseModule: mmcv.runner -> mmengine.model
- load_checkpoint: mmcv.runner -> mmengine.runner
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
import warnings

from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from mmcv.cnn import build_norm_layer
from mmengine.utils import to_2tuple
from timm.models.layers import DropPath

from mmseg.registry import MODELS


# ==================== Modal-Specific Patch Embedding ====================
class ModalSpecificPatchEmbed(nn.Module):
    """模态专用Patch Embedding - 无零填充版本"""

    def __init__(self,
                 modal_configs,
                 training_modals,
                 embed_dims=96,
                 patch_size=4,
                 norm_cfg=dict(type='LN')):
        super().__init__()
        self.modal_configs = modal_configs
        self.training_modals = set(training_modals) if training_modals else set()
        self.embed_dims = embed_dims
        self.patch_size = patch_size

        self.modal_patch_embeds = nn.ModuleDict()
        for modal_name in self.training_modals:
            if modal_name in modal_configs:
                in_ch = modal_configs[modal_name]['channels']
                self.modal_patch_embeds[modal_name] = nn.Sequential(
                    nn.Conv2d(in_ch, embed_dims,
                              kernel_size=patch_size,
                              stride=patch_size),
                    nn.LayerNorm(embed_dims, eps=1e-6)
                )

        self.register_buffer('forward_count', torch.zeros(1, dtype=torch.long))
        self._print_init_info()

    def _print_init_info(self):
        print(f"\n{'=' * 80}")
        print(f"Modal-Specific Patch Embedding (NO ZERO PADDING)")
        print(f"{'=' * 80}")
        print(f"Embed dims: {self.embed_dims}")
        print(f"Patch size: {self.patch_size}")
        print(f"Training modals: {sorted(self.training_modals)}")

        total_params = 0
        for modal_name in sorted(self.modal_patch_embeds.keys()):
            in_ch = self.modal_configs[modal_name]['channels']
            params = in_ch * self.embed_dims * self.patch_size * self.patch_size
            total_params += params
            print(f"  {modal_name:>10s}: {in_ch:>2d}ch -> {self.embed_dims:>3d}d "
                  f"({params:>6,d} params)")

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Strategy: Direct processing (NO zero-padding!)")
        print(f"{'=' * 80}\n")

    def forward(self, imgs, modal_types):
        """
        Args:
            imgs: List[Tensor] - each [C_i, H, W]
            modal_types: List[str]
        Returns:
            x: [B, H_p*W_p, embed_dims]
            hw_shape: (H_p, W_p)
        """
        self.forward_count += 1

        if self.forward_count == 1 and self.training:
            self._print_first_forward(imgs, modal_types)

        outputs = []
        hw_shape = None

        for img, modal in zip(imgs, modal_types):
            x_i = img.unsqueeze(0)

            if modal in self.modal_patch_embeds:
                conv = self.modal_patch_embeds[modal][0]
                norm = self.modal_patch_embeds[modal][1]

                out = conv(x_i)
                _, C, H, W = out.shape
                if hw_shape is None:
                    hw_shape = (H, W)

                out = out.flatten(2).transpose(1, 2)
                out = norm(out)
            else:
                if self.training:
                    raise ValueError(
                        f"Modal '{modal}' not in training modals: "
                        f"{self.training_modals}"
                    )
                else:
                    in_ch = x_i.shape[1]
                    temp_conv = nn.Conv2d(
                        in_ch, self.embed_dims,
                        kernel_size=self.patch_size,
                        stride=self.patch_size).to(x_i.device)
                    temp_norm = nn.LayerNorm(self.embed_dims).to(x_i.device)

                    nn.init.trunc_normal_(temp_conv.weight, std=0.02)
                    nn.init.constant_(temp_conv.bias, 0)

                    out = temp_conv(x_i)
                    _, C, H, W = out.shape
                    if hw_shape is None:
                        hw_shape = (H, W)
                    out = out.flatten(2).transpose(1, 2)
                    out = temp_norm(out)

            outputs.append(out)

        x = torch.cat(outputs, dim=0)
        return x, hw_shape

    def _print_first_forward(self, imgs, modal_types):
        from collections import Counter
        print(f"\n{'=' * 80}")
        print(f"Modal-Specific Patch Embedding - First Forward (NO PADDING)")
        print(f"{'=' * 80}")
        print(f"Batch size: {len(imgs)}")

        modal_counts = Counter(modal_types)
        print(f"\nBatch composition:")
        for modal, count in sorted(modal_counts.items()):
            idx = modal_types.index(modal)
            ch = imgs[idx].shape[0]
            print(f"  {modal}: {count} samples x {ch}ch (direct, no padding!)")
        print(f"{'=' * 80}\n")


class UnifiedPatchEmbed(nn.Module):
    """统一Patch Embedding - 所有模态共享一个卷积核(消融baseline)"""

    def __init__(self,
                 modal_configs,
                 training_modals,
                 embed_dims=96,
                 patch_size=4,
                 norm_cfg=dict(type='LN')):
        super().__init__()
        self.modal_configs = modal_configs
        self.training_modals = set(training_modals) if training_modals else set()
        self.embed_dims = embed_dims
        self.patch_size = patch_size

        self.max_channels = max(
            modal_configs[m]['channels'] for m in training_modals
        ) if training_modals and modal_configs else 3

        self.modal_channels = {}
        for modal_name in (training_modals or []):
            if modal_name in modal_configs:
                self.modal_channels[modal_name] = modal_configs[modal_name]['channels']

        self.unified_patch_embed = nn.Sequential(
            nn.Conv2d(self.max_channels, embed_dims,
                      kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm(embed_dims, eps=1e-6)
        )

        self.register_buffer('forward_count', torch.zeros(1, dtype=torch.long))

    def forward(self, imgs, modal_types):
        self.forward_count += 1
        outputs = []
        hw_shape = None

        for img, modal in zip(imgs, modal_types):
            x_i = img.unsqueeze(0)
            actual_ch = x_i.shape[1]

            if actual_ch < self.max_channels:
                pad_size = self.max_channels - actual_ch
                padding = torch.zeros(
                    1, pad_size, x_i.shape[2], x_i.shape[3],
                    device=x_i.device, dtype=x_i.dtype
                )
                x_i = torch.cat([x_i, padding], dim=1)

            conv = self.unified_patch_embed[0]
            norm = self.unified_patch_embed[1]

            out = conv(x_i)
            _, C, H, W = out.shape
            if hw_shape is None:
                hw_shape = (H, W)

            out = out.flatten(2).transpose(1, 2)
            out = norm(out)
            outputs.append(out)

        x = torch.cat(outputs, dim=0)
        return x, hw_shape


# ==================== MoE Components ====================
class CosineTopKGate(nn.Module):
    """Cosine similarity Gating with modal bias"""

    def __init__(self, model_dim, num_experts,
                 modal_configs=None, training_modals=None, init_t=0.5,
                 use_modal_bias=True):
        super().__init__()
        proj_dim = min(model_dim // 2, 256)

        self.temperature = nn.Parameter(
            torch.log(torch.full([1], 1.0 / init_t)),
            requires_grad=True
        )
        self.cosine_projector = nn.Linear(model_dim, proj_dim)
        self.sim_matrix = nn.Parameter(
            torch.randn(size=(proj_dim, num_experts)),
            requires_grad=True
        )
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()

        self.modal_configs = modal_configs
        self.training_modals = set(training_modals) if training_modals else set()
        self.use_modal_bias = use_modal_bias

        if modal_configs is not None and use_modal_bias:
            self.modal_bias = nn.Parameter(
                torch.zeros(len(modal_configs), num_experts),
                requires_grad=True
            )
            self.modal_name_to_idx = {
                name: i for i, name in enumerate(modal_configs.keys())
            }

        nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x, modal_types=None):
        if len(x.shape) == 4:
            B, H, W, C = x.shape
            x = x.reshape(B, -1, C).mean(dim=1)
        elif len(x.shape) == 3:
            x = x.mean(dim=1)

        logits = torch.matmul(
            F.normalize(self.cosine_projector(x), dim=1),
            F.normalize(self.sim_matrix, dim=0)
        )

        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale

        if (self.use_modal_bias and modal_types is not None
                and self.modal_configs is not None):
            modal_bias = torch.zeros_like(logits)
            for i, modal in enumerate(modal_types):
                if (modal in self.modal_name_to_idx
                        and modal in self.training_modals):
                    modal_idx = self.modal_name_to_idx[modal]
                    modal_bias[i] = self.modal_bias[modal_idx]
            logits = logits + modal_bias

        return logits


class SparseDispatcher:
    """Sparse dispatcher for MoE"""

    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = sorted_experts[index_sorted_experts[:, 1], 0]
        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())

        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        zeros = torch.zeros(
            self._gates.size(0), expert_out[-1].size(1),
            requires_grad=True, device=stitched.device
        )
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined


class SwinFFN(nn.Module):
    """Swin-style FFN (single expert)"""

    def __init__(self, in_features, hidden_features,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinMoELayer(nn.Module):
    """MoE Layer for Swin Transformer"""

    def __init__(self,
                 in_features,
                 hidden_features,
                 num_experts=8,
                 num_shared_experts=0,
                 top_k=2,
                 noisy_gating=True,
                 modal_configs=None,
                 training_modals=None,
                 use_modal_bias=True,
                 act_layer=nn.GELU,
                 drop=0.,
                 return_expert_outputs=False):
        super().__init__()
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.return_expert_outputs = return_expert_outputs

        self.experts = nn.ModuleList([
            SwinFFN(in_features, hidden_features, act_layer, drop)
            for _ in range(num_experts)
        ])

        self.gating = CosineTopKGate(
            in_features, num_experts,
            modal_configs, training_modals,
            use_modal_bias=use_modal_bias
        )

        if noisy_gating:
            self.w_noise = nn.Parameter(
                torch.zeros(in_features, num_experts),
                requires_grad=True
            )

        if num_shared_experts > 0:
            shared_hidden = hidden_features * num_shared_experts
            self.shared_experts = SwinFFN(
                in_features, shared_hidden, act_layer, drop
            )
        else:
            self.shared_experts = None

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0.0], device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def noisy_top_k_gating(self, x, modal_types=None, noise_epsilon=1e-2):
        x_pooled = x.mean(dim=1)
        clean_logits = self.gating(x_pooled, modal_types)

        if self.noisy_gating and self.training:
            raw_noise_stddev = x_pooled @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(
            min(self.top_k, self.num_experts), dim=-1
        )
        top_k_gates = self.softmax(top_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(-1, top_indices, top_k_gates)

        load = (gates > 0).sum(0)
        return gates, load

    def forward(self, x, modal_types=None, loss_coef=1e-2):
        identity = x
        B, N, C = x.shape

        gates, load = self.noisy_top_k_gating(x, modal_types)
        importance = gates.sum(0)

        balance_loss = (self.cv_squared(importance)
                        + self.cv_squared(load.float()))
        balance_loss = balance_loss * loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)

        x_for_dispatch = x.reshape(B, -1)
        expert_inputs = dispatcher.dispatch(x_for_dispatch)

        expert_outputs = []
        expert_features_list = []

        for i, expert_input in enumerate(expert_inputs):
            if expert_input.size(0) > 0:
                n_samples = expert_input.size(0)
                expert_input_reshaped = expert_input.reshape(n_samples, N, C)

                expert_out = self.experts[i](expert_input_reshaped)

                if self.return_expert_outputs and self.training:
                    expert_feat_pooled = expert_out.mean(dim=1)
                    expert_feat_4d = expert_feat_pooled.unsqueeze(-1).unsqueeze(-1)
                    expert_features_list.append(expert_feat_4d)

                expert_out = expert_out.reshape(n_samples, -1)
                expert_outputs.append(expert_out)
            else:
                expert_outputs.append(
                    torch.empty(0, N * C, device=x.device))

        y_routed = dispatcher.combine(expert_outputs)
        y_routed = y_routed.reshape(B, N, C)

        if self.shared_experts is not None:
            y_shared = self.shared_experts(identity)
        else:
            y_shared = 0

        y = y_routed + y_shared

        if (self.return_expert_outputs and self.training
                and len(expert_features_list) > 0):
            if self.shared_experts is not None:
                y_shared_pooled = y_shared.mean(dim=1)
                y_shared_4d = y_shared_pooled.unsqueeze(-1).unsqueeze(-1)
                expert_features_list.append(y_shared_4d)

            return y, balance_loss, expert_features_list
        else:
            return y, balance_loss, None


# ==================== Swin Transformer Components ====================
class WindowMSA(nn.Module):
    """Window-based Multi-head Self Attention"""

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (coords_flatten[:, :, None]
                           - coords_flatten[:, None, :])
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer(
            "relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            if B_ % nW == 0:
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
                attn = attn + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinBlockWithMoE(nn.Module):
    """Swin Transformer Block with optional MoE"""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_moe=False,
                 num_experts=8,
                 num_shared_experts=0,
                 top_k=2,
                 noisy_gating=True,
                 modal_configs=None,
                 training_modals=None,
                 use_modal_bias=True,
                 return_expert_outputs=False):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_moe = use_moe

        assert 0 <= self.shift_size < self.window_size, \
            "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowMSA(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = (DropPath(drop_path)
                          if drop_path > 0. else nn.Identity())
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        if use_moe:
            self.mlp = SwinMoELayer(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                noisy_gating=noisy_gating,
                modal_configs=modal_configs,
                training_modals=training_modals,
                use_modal_bias=use_modal_bias,
                act_layer=act_layer,
                drop=drop,
                return_expert_outputs=return_expert_outputs
            )
        else:
            self.mlp = SwinFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop
            )

        self.register_buffer("attn_mask", None)

    def forward(self, x, hw_shape, modal_types=None):
        H, W = hw_shape
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.attn_mask is None:
                self.attn_mask = self._calculate_mask(Hp, Wp).to(x.device)
            attn_mask = self.attn_mask
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        moe_loss = None
        expert_features = None

        if self.use_moe:
            mlp_out = self.mlp(self.norm2(x), modal_types)
            if len(mlp_out) == 3:
                mlp_x, moe_loss, expert_features = mlp_out
            else:
                mlp_x, moe_loss = mlp_out
            x = x + self.drop_path(mlp_x)
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, moe_loss, expert_features

    def _calculate_mask(self, H, W):
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(
            -1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class PatchMerging(nn.Module):
    """Patch Merging Layer"""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, hw_shape):
        H, W = hw_shape
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, \
            f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x, (H // 2, W // 2)


# ==================== Main Backbone ====================
@MODELS.register_module()
class MultiModalSwinMoE(BaseModule):
    """Multi-Modal Swin Transformer with MoE - MMSeg 1.x"""

    def __init__(self,
                 modal_configs=None,
                 training_modals=None,
                 pretrain_img_size=224,
                 patch_size=4,
                 embed_dims=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=[0, 1, 2, 3],
                 use_moe=True,
                 use_modal_bias=True,
                 num_experts=8,
                 num_shared_experts_config=None,
                 top_k=2,
                 noisy_gating=True,
                 MoE_Block_inds=None,
                 use_expert_diversity_loss=False,
                 use_modal_specific_stem=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.depths = depths
        self.num_heads = num_heads
        self.use_moe = use_moe
        self.use_expert_diversity_loss = use_expert_diversity_loss

        self.modal_configs = modal_configs
        if modal_configs is not None:
            if training_modals is None:
                self.training_modals = list(modal_configs.keys())
            else:
                self.training_modals = training_modals
        else:
            self.training_modals = []

        if MoE_Block_inds is None:
            MoE_Block_inds = []
            for depth in depths:
                start_idx = depth // 2
                MoE_Block_inds.append(list(range(start_idx, depth)))
        self.MoE_Block_inds = MoE_Block_inds

        if num_shared_experts_config is None:
            num_shared_experts_config = {0: 0, 1: 0, 2: 2, 3: 1}
        self.num_shared_experts_config = num_shared_experts_config

        self.use_modal_specific_stem = use_modal_specific_stem
        if use_modal_specific_stem:
            self.patch_embed = ModalSpecificPatchEmbed(
                modal_configs=modal_configs if modal_configs else {},
                training_modals=self.training_modals,
                embed_dims=embed_dims,
                patch_size=patch_size,
                norm_cfg=dict(type='LN') if patch_norm else None
            )
        else:
            self.patch_embed = UnifiedPatchEmbed(
                modal_configs=modal_configs if modal_configs else {},
                training_modals=self.training_modals,
                embed_dims=embed_dims,
                patch_size=patch_size,
                norm_cfg=dict(type='LN') if patch_norm else None
            )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        num_features = [int(embed_dims * 2 ** i)
                        for i in range(self.num_layers)]

        for i_stage in range(self.num_layers):
            stage_blocks = []
            dim = num_features[i_stage]
            num_shared = num_shared_experts_config.get(i_stage, 0)

            for i_block in range(depths[i_stage]):
                use_moe_block = (use_moe
                                 and (i_block in MoE_Block_inds[i_stage]))

                block = SwinBlockWithMoE(
                    dim=dim,
                    num_heads=num_heads[i_stage],
                    window_size=window_size,
                    shift_size=(0 if (i_block % 2 == 0)
                                else window_size // 2),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_stage]) + i_block],
                    norm_layer=norm_layer,
                    use_moe=use_moe_block,
                    num_experts=num_experts,
                    num_shared_experts=(num_shared
                                       if use_moe_block else 0),
                    top_k=top_k,
                    noisy_gating=noisy_gating,
                    modal_configs=(modal_configs
                                  if use_moe_block else None),
                    training_modals=(self.training_modals
                                    if use_moe_block else None),
                    use_modal_bias=use_modal_bias,
                    return_expert_outputs=use_expert_diversity_loss
                )
                stage_blocks.append(block)

            if i_stage < self.num_layers - 1:
                downsample = PatchMerging(dim=dim, norm_layer=norm_layer)
            else:
                downsample = None

            self.stages.append(nn.ModuleDict({
                'blocks': nn.ModuleList(stage_blocks),
                'downsample': downsample
            }))

        for i in out_indices:
            layer = norm_layer(num_features[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        self.num_features = num_features
        self.pretrained = pretrained
        self._init_weights()
        self._print_architecture_info()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        if self.pretrained is not None:
            print(f'Loading pretrained Swin weights from {self.pretrained}')
            load_checkpoint(
                self, self.pretrained,
                map_location='cpu',
                strict=False
            )

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        keys_to_remove = [k for k in state_dict.keys() if 'attn_mask' in k]
        for k in keys_to_remove:
            state_dict.pop(k)
        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        attn_mask_keys = [k for k in state_dict.keys() if 'attn_mask' in k]
        for k in attn_mask_keys:
            state_dict.pop(k)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def _print_architecture_info(self):
        print(f"\n{'=' * 80}")
        print(f"Multi-Modal Swin Transformer with MoE")
        print(f"{'=' * 80}")
        print(f"Training modals: {self.training_modals}")
        print(f"Embed dims: {self.embed_dims}")
        print(f"Depths: {self.depths}")
        print(f"Use MoE: {self.use_moe}")

        print(f"\nShared Experts:")
        for i in range(self.num_layers):
            num_moe = len(self.MoE_Block_inds[i])
            num_shared = self.num_shared_experts_config.get(i, 0)
            print(f"  Stage {i}: {self.depths[i]} blocks, "
                  f"{num_moe} MoE blocks, {num_shared} shared experts")

        print(f"\nOutput indices: {self.out_indices}")
        print(f"Output features: "
              f"{[self.num_features[i] for i in self.out_indices]}")
        print(f"{'=' * 80}\n")

    def forward(self, imgs, modal_types=None, **kwargs):
        """
        Args:
            imgs: List[Tensor] - each [C_i, H, W]
            modal_types: List[str]
        Returns:
            tuple of features, moe_balance_loss, expert_features
        """
        B = len(imgs)

        if modal_types is None:
            modal_types = ['rgb'] * B

        x, hw_shape = self.patch_embed(imgs, modal_types)
        x = self.pos_drop(x)

        outs = []
        moe_balance_losses = []
        all_expert_features = []

        for i_stage, stage_dict in enumerate(self.stages):
            blocks = stage_dict['blocks']
            downsample = stage_dict['downsample']

            for block in blocks:
                x, moe_loss, expert_features = block(
                    x, hw_shape, modal_types)

                if moe_loss is not None:
                    moe_balance_losses.append(moe_loss)

                if (expert_features is not None
                        and self.use_expert_diversity_loss
                        and i_stage == self.num_layers - 1):
                    all_expert_features.extend(expert_features)

            if i_stage in self.out_indices:
                norm_layer = getattr(self, f'norm{i_stage}')
                x_out = norm_layer(x)

                H, W = hw_shape
                x_out = x_out.view(B, H, W, -1).permute(
                    0, 3, 1, 2).contiguous()
                outs.append(x_out)

            if downsample is not None:
                x, hw_shape = downsample(x, hw_shape)

        avg_balance_loss = None
        if len(moe_balance_losses) > 0:
            avg_balance_loss = (sum(moe_balance_losses)
                                / len(moe_balance_losses))

        return tuple(outs), avg_balance_loss, all_expert_features
