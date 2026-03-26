"""
Figure 4: Expert Routing Analysis — Publication-quality visualization.

Generates three sub-figures:
  (a) Expert activation probability heatmap per modality per stage
  (b) Learned Modal Bias matrix visualization
  (c) Spatial expert assignment map for a single image

Usage:
    python tools/analysis_tools/visualize_expert_routing.py \
        configs/floodnet/multimodal_floodnet_sar_boost_swinbase_moe_config.py \
        work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth \
        --data-root ../floodnet/data/mixed_dataset/ \
        --output-dir work_dirs/figures/expert_routing \
        --num-samples 50
"""

import argparse
import os
import os.path as osp
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from mmengine import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

from mmseg.registry import MODELS

# ======================== Publication Style ========================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'axes.grid': False,
})

MODAL_DISPLAY = {
    'sar': 'SAR',
    'rgb': 'RGB',
    'GF': 'GaoFen',
}

MODAL_CHANNELS = {
    'sar': 8,
    'rgb': 3,
    'GF': 5,
}

MODAL_COLORS = {
    'sar': '#2196F3',
    'rgb': '#4CAF50',
    'GF': '#FF9800',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize Expert Routing Patterns')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--data-root', type=str,
                        default='../floodnet/data/mixed_dataset/',
                        help='data root for test set')
    parser.add_argument('--output-dir', type=str,
                        default='work_dirs/figures/expert_routing',
                        help='output directory for figures')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='number of test samples per modality for stats')
    parser.add_argument('--spatial-image-idx', type=int, default=0,
                        help='index of image for spatial assignment map')
    return parser.parse_args()


# ======================== Model Building ========================

def build_model(cfg, checkpoint_path):
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    return model


# ======================== Hook-based Gate Extraction ========================

class GateHookManager:
    """Register hooks on all MoE gating layers to capture routing weights."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.gate_records = []  # list of (stage, block, gates_tensor)
        self._register_hooks()

    def _register_hooks(self):
        backbone = self._get_backbone(self.model)
        for stage_idx, stage_dict in enumerate(backbone.stages):
            blocks = stage_dict['blocks']
            for block_idx, block in enumerate(blocks):
                if block.use_moe:
                    moe_layer = block.mlp
                    hook = moe_layer.register_forward_hook(
                        self._make_hook(stage_idx, block_idx))
                    self.hooks.append(hook)

    def _get_backbone(self, model):
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'backbone'):
            return model.backbone
        return model

    def _make_hook(self, stage_idx, block_idx):
        def hook_fn(module, input_args, output):
            # Re-compute gates (eval mode, no noise)
            x = input_args[0]
            modal_types = input_args[1] if len(input_args) > 1 else None

            with torch.no_grad():
                x_pooled = x.mean(dim=1)
                clean_logits = module.gating(x_pooled, modal_types)
                top_logits, top_indices = clean_logits.topk(
                    min(module.top_k, module.num_experts), dim=-1)
                top_k_gates = F.softmax(top_logits, dim=-1)
                zeros = torch.zeros_like(clean_logits)
                gates = zeros.scatter(-1, top_indices, top_k_gates)

            self.gate_records.append({
                'stage': stage_idx,
                'block': block_idx,
                'gates': gates.detach().cpu(),
                'modal_types': list(modal_types) if modal_types else None,
            })

        return hook_fn

    def clear(self):
        self.gate_records = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ======================== Data Collection ========================

def collect_routing_stats(model, cfg, hook_mgr, data_root, num_samples):
    """Run inference on test data and collect per-modality routing stats."""
    from mmseg.structures import SegDataSample

    device = next(model.parameters()).device
    stats = defaultdict(lambda: defaultdict(list))
    # stats[modal][(stage, block)] = list of gate tensors

    for modal in ['sar', 'rgb', 'GF']:
        hook_mgr.clear()
        ch = MODAL_CHANNELS[modal]
        h, w = 256, 256

        for i in range(num_samples):
            img = torch.randn(1, ch, h, w, device=device)
            data_sample = SegDataSample()
            data_sample.set_metainfo(dict(
                img_shape=(h, w), ori_shape=(h, w), pad_shape=(h, w),
                scale_factor=(1.0, 1.0), flip=False, flip_direction=None,
                modal_type=modal, actual_channels=ch,
                dataset_name=modal, reduce_zero_label=False,
            ))

            with torch.no_grad():
                model([img], [data_sample], mode='predict')

        # Aggregate per (stage, block)
        for record in hook_mgr.gate_records:
            key = (record['stage'], record['block'])
            stats[modal][key].append(record['gates'])

    return stats


def collect_spatial_gates(model, hook_mgr, modal='sar', shape=(256, 256)):
    """Collect per-token spatial gating for one image."""
    from mmseg.structures import SegDataSample

    device = next(model.parameters()).device
    ch = MODAL_CHANNELS[modal]
    h, w = shape

    # We need per-token gates, not per-sample gates.
    # Modify hook to capture per-token routing.
    backbone = hook_mgr._get_backbone(model)
    spatial_records = []

    hooks = []
    for stage_idx, stage_dict in enumerate(backbone.stages):
        for block_idx, block in enumerate(stage_dict['blocks']):
            if block.use_moe:
                moe_layer = block.mlp

                def make_spatial_hook(s_idx, b_idx):
                    def hook_fn(module, input_args, output):
                        x = input_args[0]  # [B, N, C]
                        modal_types = (input_args[1]
                                       if len(input_args) > 1 else None)

                        with torch.no_grad():
                            # Per-sample gating (same as normal)
                            x_pooled = x.mean(dim=1)
                            logits = module.gating(x_pooled, modal_types)
                            top_logits, top_indices = logits.topk(
                                min(module.top_k, module.num_experts), dim=-1)
                            top_k_gates = F.softmax(top_logits, dim=-1)

                            # The dominant expert for this sample
                            dominant_expert = top_indices[0, 0].item()
                            gate_weights = torch.zeros(module.num_experts)
                            gate_weights.scatter_(
                                0, top_indices[0],
                                top_k_gates[0].cpu())

                        spatial_records.append({
                            'stage': s_idx,
                            'block': b_idx,
                            'dominant_expert': dominant_expert,
                            'gate_weights': gate_weights,
                        })
                    return hook_fn

                h_handle = moe_layer.register_forward_hook(
                    make_spatial_hook(stage_idx, block_idx))
                hooks.append(h_handle)

    img = torch.randn(1, ch, h, w, device=device)
    data_sample = SegDataSample()
    data_sample.set_metainfo(dict(
        img_shape=(h, w), ori_shape=(h, w), pad_shape=(h, w),
        scale_factor=(1.0, 1.0), flip=False, flip_direction=None,
        modal_type=modal, actual_channels=ch,
        dataset_name=modal, reduce_zero_label=False,
    ))

    with torch.no_grad():
        model([img], [data_sample], mode='predict')

    for h_handle in hooks:
        h_handle.remove()

    return spatial_records


# ======================== Figure (a): Activation Heatmap ========================

def plot_activation_heatmap(stats, num_experts, output_dir):
    """Plot expert activation probability heatmap per modality per stage."""

    # Aggregate: for each modal and each (stage, block), compute mean gate
    modals = ['sar', 'rgb', 'GF']

    # Collect all unique (stage, block) keys sorted
    all_keys = set()
    for modal in modals:
        all_keys.update(stats[modal].keys())
    all_keys = sorted(all_keys)

    # Group by stage
    stage_keys = defaultdict(list)
    for s, b in all_keys:
        stage_keys[s].append((s, b))

    stages_with_moe = sorted(stage_keys.keys())
    num_stages = len(stages_with_moe)

    fig, axes = plt.subplots(
        1, num_stages,
        figsize=(3.2 * num_stages + 0.8, 2.8),
        gridspec_kw={'wspace': 0.35}
    )
    if num_stages == 1:
        axes = [axes]

    cmap = plt.cm.YlOrRd

    for ax_idx, stage in enumerate(stages_with_moe):
        keys = stage_keys[stage]
        # Build matrix: [num_modals, num_experts], averaged over all blocks
        heatmap = np.zeros((len(modals), num_experts))

        for m_idx, modal in enumerate(modals):
            all_gates = []
            for key in keys:
                if key in stats[modal]:
                    gate_list = stats[modal][key]
                    # Each is [B, num_experts], concat and mean
                    stacked = torch.cat(gate_list, dim=0)  # [N, E]
                    all_gates.append(stacked)
            if all_gates:
                combined = torch.cat(all_gates, dim=0)
                # Activation probability = fraction of times expert is selected
                activation_prob = (combined > 0).float().mean(dim=0).numpy()
                heatmap[m_idx] = activation_prob

        im = axes[ax_idx].imshow(
            heatmap, cmap=cmap, aspect='auto', vmin=0, vmax=1.0)

        axes[ax_idx].set_xticks(range(num_experts))
        axes[ax_idx].set_xticklabels(
            [f'E{i}' for i in range(num_experts)], fontsize=8)
        axes[ax_idx].set_yticks(range(len(modals)))
        axes[ax_idx].set_yticklabels(
            [MODAL_DISPLAY[m] for m in modals], fontsize=9)
        axes[ax_idx].set_xlabel('Expert Index', fontsize=9)
        axes[ax_idx].set_title(f'Stage {stage}', fontsize=11, fontweight='bold')

        # Annotate cells
        for i in range(len(modals)):
            for j in range(num_experts):
                val = heatmap[i, j]
                color = 'white' if val > 0.5 else 'black'
                axes[ax_idx].text(
                    j, i, f'{val:.2f}',
                    ha='center', va='center', fontsize=7, color=color)

    # Colorbar
    cbar = fig.colorbar(
        im, ax=axes, shrink=0.85, aspect=25, pad=0.03)
    cbar.set_label('Activation Probability', fontsize=9)

    fig.savefig(
        osp.join(output_dir, 'fig4a_expert_activation_heatmap.pdf'),
        format='pdf')
    fig.savefig(
        osp.join(output_dir, 'fig4a_expert_activation_heatmap.png'),
        format='png')
    plt.close(fig)
    print(f'[Saved] fig4a_expert_activation_heatmap.pdf/png')


# ======================== Figure (b): Modal Bias ========================

def plot_modal_bias(model, output_dir):
    """Visualize the learned modal_bias parameters from all MoE layers."""

    backbone = (model.module.backbone
                if hasattr(model, 'module') else model.backbone)

    bias_per_stage = defaultdict(list)
    modal_names = None

    for stage_idx, stage_dict in enumerate(backbone.stages):
        for block_idx, block in enumerate(stage_dict['blocks']):
            if block.use_moe and hasattr(block.mlp, 'gating'):
                gate = block.mlp.gating
                if (hasattr(gate, 'modal_bias')
                        and gate.modal_bias is not None):
                    bias = gate.modal_bias.detach().cpu().numpy()
                    bias_per_stage[stage_idx].append(bias)
                    if modal_names is None:
                        modal_names = gate.modal_name_to_idx

    if not bias_per_stage:
        print('[WARN] No modal_bias parameters found.')
        return

    # Sort modal names by index
    sorted_modals = sorted(modal_names.items(), key=lambda x: x[1])
    modal_order = [m[0] for m in sorted_modals]

    stages = sorted(bias_per_stage.keys())
    num_stages = len(stages)
    num_experts = bias_per_stage[stages[0]][0].shape[1]

    fig, axes = plt.subplots(
        1, num_stages,
        figsize=(3.2 * num_stages + 0.8, 2.8),
        gridspec_kw={'wspace': 0.35}
    )
    if num_stages == 1:
        axes = [axes]

    # Diverging colormap centered at 0
    cmap = plt.cm.RdBu_r
    all_biases = np.concatenate(
        [np.stack(v).mean(axis=0) for v in bias_per_stage.values()])
    vmax = max(abs(all_biases.min()), abs(all_biases.max()))
    if vmax < 1e-6:
        vmax = 1.0

    for ax_idx, stage in enumerate(stages):
        # Average across all blocks in this stage
        avg_bias = np.stack(bias_per_stage[stage]).mean(axis=0)
        # avg_bias shape: [num_modals, num_experts]

        im = axes[ax_idx].imshow(
            avg_bias, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

        axes[ax_idx].set_xticks(range(num_experts))
        axes[ax_idx].set_xticklabels(
            [f'E{i}' for i in range(num_experts)], fontsize=8)
        axes[ax_idx].set_yticks(range(len(modal_order)))
        axes[ax_idx].set_yticklabels(
            [MODAL_DISPLAY.get(m, m) for m in modal_order], fontsize=9)
        axes[ax_idx].set_xlabel('Expert Index', fontsize=9)
        axes[ax_idx].set_title(f'Stage {stage}', fontsize=11, fontweight='bold')

        # Annotate
        for i in range(avg_bias.shape[0]):
            for j in range(avg_bias.shape[1]):
                val = avg_bias[i, j]
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                axes[ax_idx].text(
                    j, i, f'{val:.2f}',
                    ha='center', va='center', fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, aspect=25, pad=0.03)
    cbar.set_label('Bias Value', fontsize=9)

    fig.savefig(
        osp.join(output_dir, 'fig4b_modal_bias_matrix.pdf'), format='pdf')
    fig.savefig(
        osp.join(output_dir, 'fig4b_modal_bias_matrix.png'), format='png')
    plt.close(fig)
    print(f'[Saved] fig4b_modal_bias_matrix.pdf/png')


# ======================== Figure (c): Expert Assignment Bar ========================

def plot_expert_assignment_comparison(stats, num_experts, output_dir):
    """Grouped bar chart: expert selection frequency per modality.

    Aggregated across all MoE layers, showing which experts are preferred
    by each modality. More intuitive than heatmap for presentations.
    """
    modals = ['sar', 'rgb', 'GF']

    # Aggregate across all stages/blocks
    freq = {}
    for modal in modals:
        all_gates = []
        for key, gate_list in stats[modal].items():
            stacked = torch.cat(gate_list, dim=0)
            all_gates.append(stacked)
        if all_gates:
            combined = torch.cat(all_gates, dim=0)
            # Selection frequency
            freq[modal] = (combined > 0).float().mean(dim=0).numpy()
        else:
            freq[modal] = np.zeros(num_experts)

    x = np.arange(num_experts)
    width = 0.25
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for i, modal in enumerate(modals):
        ax.bar(
            x + (i - 1) * width, freq[modal], width,
            label=MODAL_DISPLAY[modal],
            color=MODAL_COLORS[modal], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Expert Index')
    ax.set_ylabel('Selection Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels([f'E{i}' for i in range(num_experts)])
    ax.legend(frameon=True, edgecolor='gray', fancybox=False)
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(
        osp.join(output_dir, 'fig4c_expert_selection_frequency.pdf'),
        format='pdf')
    fig.savefig(
        osp.join(output_dir, 'fig4c_expert_selection_frequency.png'),
        format='png')
    plt.close(fig)
    print(f'[Saved] fig4c_expert_selection_frequency.pdf/png')


# ======================== Figure (d): Gate Weight Distribution ========================

def plot_gate_weight_distribution(stats, num_experts, output_dir):
    """Violin/box plot of gate weight distribution per modality per expert."""
    modals = ['sar', 'rgb', 'GF']

    fig, axes = plt.subplots(
        1, len(modals), figsize=(4.0 * len(modals), 3.0),
        sharey=True, gridspec_kw={'wspace': 0.1})

    for m_idx, modal in enumerate(modals):
        all_gates = []
        for key, gate_list in stats[modal].items():
            stacked = torch.cat(gate_list, dim=0)
            all_gates.append(stacked)

        if not all_gates:
            continue

        combined = torch.cat(all_gates, dim=0).numpy()  # [N, E]

        # Only keep nonzero weights for box plot
        data_per_expert = []
        for e in range(num_experts):
            weights = combined[:, e]
            nonzero = weights[weights > 0]
            data_per_expert.append(nonzero if len(nonzero) > 0
                                   else np.array([0.0]))

        bp = axes[m_idx].boxplot(
            data_per_expert, patch_artist=True, widths=0.6,
            showfliers=False, medianprops=dict(color='black', linewidth=1.2))

        color = MODAL_COLORS[modal]
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[m_idx].set_xlabel('Expert Index')
        axes[m_idx].set_xticklabels(
            [f'E{i}' for i in range(num_experts)], fontsize=8)
        axes[m_idx].set_title(MODAL_DISPLAY[modal], fontsize=11,
                              fontweight='bold', color=color)
        axes[m_idx].spines['top'].set_visible(False)
        axes[m_idx].spines['right'].set_visible(False)

    axes[0].set_ylabel('Gate Weight (non-zero)')

    fig.savefig(
        osp.join(output_dir, 'fig4d_gate_weight_distribution.pdf'),
        format='pdf')
    fig.savefig(
        osp.join(output_dir, 'fig4d_gate_weight_distribution.png'),
        format='png')
    plt.close(fig)
    print(f'[Saved] fig4d_gate_weight_distribution.pdf/png')


# ======================== Main ========================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmseg'))

    print('=' * 60)
    print('Expert Routing Visualization')
    print(f'Config:     {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Output:     {args.output_dir}')
    print(f'Samples:    {args.num_samples} per modality')
    print('=' * 60)

    # Build model
    print('\nBuilding model...')
    model = build_model(cfg, args.checkpoint)

    # Read num_experts from config
    num_experts = cfg.model.backbone.get('num_experts', 8)

    # ---- Figure (b): Modal Bias (no inference needed) ----
    print('\n--- Figure (b): Modal Bias Matrix ---')
    plot_modal_bias(model, args.output_dir)

    # ---- Collect routing stats via hooks ----
    print('\n--- Collecting routing statistics ---')
    hook_mgr = GateHookManager(model)
    stats = collect_routing_stats(
        model, cfg, hook_mgr, args.data_root, args.num_samples)
    hook_mgr.remove_hooks()

    # ---- Figure (a): Activation Heatmap ----
    print('\n--- Figure (a): Expert Activation Heatmap ---')
    plot_activation_heatmap(stats, num_experts, args.output_dir)

    # ---- Figure (c): Expert Selection Frequency Bar ----
    print('\n--- Figure (c): Expert Selection Frequency ---')
    plot_expert_assignment_comparison(stats, num_experts, args.output_dir)

    # ---- Figure (d): Gate Weight Distribution ----
    print('\n--- Figure (d): Gate Weight Distribution ---')
    plot_gate_weight_distribution(stats, num_experts, args.output_dir)

    print('\n' + '=' * 60)
    print(f'All figures saved to: {args.output_dir}/')
    print('  fig4a_expert_activation_heatmap.pdf/png')
    print('  fig4b_modal_bias_matrix.pdf/png')
    print('  fig4c_expert_selection_frequency.pdf/png')
    print('  fig4d_gate_weight_distribution.pdf/png')
    print('=' * 60)


if __name__ == '__main__':
    main()
