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
from mmengine import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
    HAS_MMENGINE_ANALYSIS = True
except ImportError:
    HAS_MMENGINE_ANALYSIS = False


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
    img = torch.randn(1, channels, h, w, device=device)

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


def measure_flops_fvcore(model, modal, shape):
    """Measure FLOPs using fvcore (more reliable for custom models)."""
    device = next(model.parameters()).device
    img, data_sample = make_dummy_input(modal, shape, device)

    # fvcore expects a callable; wrap model forward
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, imgs, data_samples):
            return self.model(imgs, data_samples, mode='predict')

    wrapper = ModelWrapper(model)
    wrapper.eval()

    try:
        flop_analysis = FlopCountAnalysis(wrapper, ([img], [data_sample]))
        flop_analysis.unsupported_ops_warnings(False)
        flop_analysis.uncalled_modules_warnings(False)
        flops = flop_analysis.total()
        return flops
    except Exception as e:
        print(f"  [fvcore WARN] {e}")
        return None


def measure_flops_mmengine(model, modal, shape):
    """Measure FLOPs using mmengine analysis."""
    device = next(model.parameters()).device
    img, data_sample = make_dummy_input(modal, shape, device)

    try:
        outputs = get_model_complexity_info(
            model,
            input_shape=None,
            inputs=[img],
            data_samples=[data_sample],
            show_table=False,
            show_arch=False,
        )
        return outputs['flops'], outputs['params']
    except Exception as e:
        print(f"  [mmengine WARN] {e}")
        return None, None


def format_flops(flops):
    """Format FLOPs to human readable string."""
    if flops is None:
        return 'N/A'
    if flops >= 1e12:
        return f'{flops / 1e12:.2f} T'
    elif flops >= 1e9:
        return f'{flops / 1e9:.2f} G'
    elif flops >= 1e6:
        return f'{flops / 1e6:.2f} M'
    else:
        return f'{flops:.0f}'


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
    print('FLOPs Measurement')
    print('=' * 60)

    for modal in args.modals:
        ch = MODAL_CHANNELS[modal]
        print(f'\n[{modal.upper()}] input: ({ch}, {args.shape[0]}, {args.shape[1]})')

        flops_val = None

        # Try fvcore first
        if HAS_FVCORE:
            print('  Method: fvcore')
            flops_val = measure_flops_fvcore(model, modal, args.shape)
            if flops_val is not None:
                print(f'  FLOPs: {format_flops(flops_val)}')

        # Try mmengine
        if HAS_MMENGINE_ANALYSIS and flops_val is None:
            print('  Method: mmengine')
            flops_mm, params_mm = measure_flops_mmengine(model, modal, args.shape)
            if flops_mm is not None:
                print(f'  FLOPs:  {_format_size(flops_mm)}')
                print(f'  Params: {_format_size(params_mm)}')

        if not HAS_FVCORE and not HAS_MMENGINE_ANALYSIS:
            print('  [ERROR] Neither fvcore nor mmengine.analysis available.')
            print('  Install fvcore: pip install fvcore')

    # ======================== FPS ========================
    print()
    print('=' * 60)
    print('FPS Measurement')
    print(f'  Warmup: {args.num_warmup} iters')
    print(f'  Timed:  {args.num_iters} iters x {args.repeat_times} runs')
    print('=' * 60)

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
        print(f'  >> Average: {avg_fps:.2f} ± {std_fps:.2f} img/s')

    # ======================== Summary ========================
    print()
    print('=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'Params: {format_params(total_params)}')
    print(f'Input:  {args.shape[0]} x {args.shape[1]}')
    print()


if __name__ == '__main__':
    main()
