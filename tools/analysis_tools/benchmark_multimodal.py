"""
Compute Inference FPS and FLOPs for Multi-Modal Swin-MoE model.

Usage:
    python tools/analysis_tools/benchmark_multimodal.py \
        configs/floodnet/multimodal_floodnet_sar_boost_swinbase_moe_config.py \
        work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth \
        --shape 256 256 \
        --repeat-times 3 \
        --num-iters 200
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from mmengine import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


MODAL_CHANNELS = {
    'sar': 8,
    'rgb': 3,
    'GF': 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark Multi-Modal Segmentor (FPS + FLOPs)')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--shape', type=int, nargs=2, default=[256, 256],
                        help='input H W (default: 256 256)')
    parser.add_argument('--repeat-times', type=int, default=3,
                        help='number of FPS measurement runs')
    parser.add_argument('--num-iters', type=int, default=200,
                        help='iterations per FPS run')
    parser.add_argument('--num-warmup', type=int, default=10,
                        help='warmup iterations before timing')
    parser.add_argument('--modals', nargs='+', default=['sar', 'rgb', 'GF'],
                        help='modalities to benchmark')
    return parser.parse_args()


def build_model(cfg, checkpoint_path):
    """Build model and load checkpoint."""
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)

    load_checkpoint(model, checkpoint_path, map_location='cpu')

    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    return model


def make_dummy_input(modal, shape, device='cuda'):
    """Create a dummy input mimicking the data preprocessor output."""
    h, w = shape
    channels = MODAL_CHANNELS[modal]
    img = torch.randn(channels, h, w, device=device)

    data_sample = SegDataSample()
    data_sample.set_metainfo(dict(
        img_shape=(h, w),
        ori_shape=(h, w),
        pad_shape=(h, w),
        scale_factor=(1.0, 1.0),
        flip=False,
        flip_direction=None,
        modal_type=modal,
        actual_channels=channels,
        dataset_name=modal,
        reduce_zero_label=False,
    ))

    return img, data_sample


def measure_fps(model, modal, shape, num_iters=200, num_warmup=10):
    """Measure inference FPS for a single modality."""
    device = next(model.parameters()).device
    img, data_sample = make_dummy_input(modal, shape, device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model([img], [data_sample], mode='predict')

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed iterations
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            model([img], [data_sample], mode='predict')

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    fps = num_iters / elapsed
    return fps


class BackboneDecoderWrapper(nn.Module):
    """Wrapper that runs backbone + decode_head forward only.

    Avoids predict/inference path so FLOPs tools (fvcore/thop)
    don't encounter SegDataSample or slide_inference logic.
    """

    def __init__(self, model, modal):
        super().__init__()
        self.backbone = model.backbone
        self.modal = modal
        # Get the correct decode head
        if hasattr(model, 'decode_heads') and modal in model.decode_heads:
            self.decode_head = model.decode_heads[modal]
        elif hasattr(model, '_shared_decode_head'):
            self.decode_head = model._shared_decode_head
        else:
            self.decode_head = None

    def forward(self, x):
        """x: (1, C, H, W) tensor for a single modality."""
        imgs_list = [x[0]]  # list of (C, H, W)
        modal_types = [self.modal]
        features, _, _ = self.backbone(imgs_list, modal_types)
        # features is a tuple of multi-scale tensors
        if self.decode_head is not None:
            out = self.decode_head(features)
        return out


def measure_flops_manual(model, modal, shape):
    """Measure FLOPs by manually counting ops in backbone + decoder.

    Uses a forward hook approach to count multiply-accumulate operations.
    """
    device = next(model.parameters()).device
    h, w = shape
    channels = MODAL_CHANNELS[modal]
    img = torch.randn(1, channels, h, w, device=device)

    wrapper = BackboneDecoderWrapper(model, modal)
    wrapper.eval()

    total_flops = 0

    def count_conv2d(m, inp, out):
        nonlocal total_flops
        x = inp[0]
        batch = x.shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        kernel_ops = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        total_flops += batch * m.out_channels * out_h * out_w * kernel_ops

    def count_linear(m, inp, out):
        nonlocal total_flops
        x = inp[0]
        batch_size = x.numel() // m.in_features
        total_flops += batch_size * m.in_features * m.out_features

    def count_layernorm(m, inp, out):
        nonlocal total_flops
        total_flops += inp[0].numel() * 2  # mean + variance

    def count_gelu(m, inp, out):
        nonlocal total_flops
        total_flops += inp[0].numel() * 4  # approximate

    def count_bn(m, inp, out):
        nonlocal total_flops
        total_flops += inp[0].numel() * 2

    hooks = []
    for m in wrapper.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(count_conv2d))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(count_linear))
        elif isinstance(m, nn.LayerNorm):
            hooks.append(m.register_forward_hook(count_layernorm))
        elif isinstance(m, nn.GELU):
            hooks.append(m.register_forward_hook(count_gelu))
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            hooks.append(m.register_forward_hook(count_bn))

    with torch.no_grad():
        try:
            wrapper(img)
        except Exception as e:
            print(f'  [manual WARN] forward failed: {e}')
            for h in hooks:
                h.remove()
            return None

    for h in hooks:
        h.remove()

    return total_flops


def measure_flops_fvcore(model, modal, shape):
    """Measure FLOPs using fvcore on backbone + decoder wrapper."""
    device = next(model.parameters()).device
    h, w = shape
    channels = MODAL_CHANNELS[modal]
    img = torch.randn(1, channels, h, w, device=device)

    wrapper = BackboneDecoderWrapper(model, modal)
    wrapper.eval()

    try:
        flop_analysis = FlopCountAnalysis(wrapper, (img,))
        flop_analysis.unsupported_ops_warnings(False)
        flop_analysis.uncalled_modules_warnings(False)
        flops = flop_analysis.total()
        return flops
    except Exception as e:
        print(f'  [fvcore WARN] {e}')
        return None


def format_flops(flops):
    """Format FLOPs to human readable string."""
    if flops is None:
        return 'N/A'
    if flops >= 1e12:
        return f'{flops / 1e12:.2f} TFLOPs'
    elif flops >= 1e9:
        return f'{flops / 1e9:.2f} GFLOPs'
    elif flops >= 1e6:
        return f'{flops / 1e6:.2f} MFLOPs'
    else:
        return f'{flops:.0f} FLOPs'


def format_params(params):
    """Format parameter count."""
    if params >= 1e6:
        return f'{params / 1e6:.2f} M'
    elif params >= 1e3:
        return f'{params / 1e3:.2f} K'
    else:
        return f'{params:.0f}'


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmseg'))

    print('=' * 60)
    print('Multi-Modal Model Benchmark')
    print(f'Config:     {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Input size: {args.shape[0]} x {args.shape[1]}')
    print(f'Modalities: {args.modals}')
    print('=' * 60)
    print()

    # Build model
    print('Building model...')
    model = build_model(cfg, args.checkpoint)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params:     {format_params(total_params)}')
    print(f'Trainable params: {format_params(trainable_params)}')
    print()

    # ======================== FLOPs ========================
    print('=' * 60)
    print('FLOPs Measurement (backbone + decode_head)')
    print('=' * 60)

    flops_results = {}
    for modal in args.modals:
        ch = MODAL_CHANNELS[modal]
        print(f'\n[{modal.upper()}] input: ({ch}, {args.shape[0]}, {args.shape[1]})')

        flops_val = None

        # Try fvcore first (wraps backbone+decoder only, no SegDataSample)
        if HAS_FVCORE:
            print('  Method: fvcore')
            flops_val = measure_flops_fvcore(model, modal, args.shape)
            if flops_val is not None:
                print(f'  FLOPs: {format_flops(flops_val)}')

        # Fallback: manual hook-based counting
        if flops_val is None:
            print('  Method: manual (hook-based)')
            flops_val = measure_flops_manual(model, modal, args.shape)
            if flops_val is not None:
                print(f'  FLOPs: {format_flops(flops_val)}')
            else:
                print('  FLOPs: N/A')

        flops_results[modal] = flops_val

    # ======================== FPS ========================
    print()
    print('=' * 60)
    print('FPS Measurement (full predict pipeline)')
    print(f'  Warmup: {args.num_warmup} iters')
    print(f'  Timed:  {args.num_iters} iters x {args.repeat_times} runs')
    print('=' * 60)

    fps_results = {}
    for modal in args.modals:
        ch = MODAL_CHANNELS[modal]
        print(f'\n[{modal.upper()}] input: ({ch}, {args.shape[0]}, {args.shape[1]})')

        fps_list = []
        for run_idx in range(args.repeat_times):
            fps = measure_fps(model, modal, args.shape,
                              num_iters=args.num_iters,
                              num_warmup=args.num_warmup)
            fps_list.append(fps)
            print(f'  Run {run_idx + 1}: {fps:.2f} img/s')

        avg_fps = np.mean(fps_list)
        std_fps = np.std(fps_list)
        print(f'  >> Average: {avg_fps:.2f} +/- {std_fps:.2f} img/s')
        fps_results[modal] = (avg_fps, std_fps)

    # ======================== Summary ========================
    print()
    print('=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'{"Modality":<10} {"Channels":<10} {"FLOPs":<18} {"FPS (avg)":>15}')
    print('-' * 55)
    for modal in args.modals:
        ch = MODAL_CHANNELS[modal]
        flops_str = format_flops(flops_results.get(modal))
        fps_avg, fps_std = fps_results.get(modal, (0, 0))
        print(f'{modal:<10} {ch:<10} {flops_str:<18} {fps_avg:>10.2f} img/s')
    print('-' * 55)
    print(f'Params: {format_params(total_params)}')
    print(f'Input:  {args.shape[0]} x {args.shape[1]}')
    print()


if __name__ == '__main__':
    main()
