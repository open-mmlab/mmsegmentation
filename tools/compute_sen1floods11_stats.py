"""Compute channel-wise mean / std for Sen1Floods11 S1Hand or S2Hand.

The tool walks ``<data_root>/<subdir>/*.tif`` (default: S1Hand or S2Hand)
and ignores pixels that are flagged as nodata in the matching
``LabelHand`` file (label value == -1). Pass the resulting mean / std
arrays into ``MultiModalNormalize.NORM_CONFIGS`` under the ``s1`` / ``s2``
entries in ``mmseg/datasets/transforms/multimodal_pipelines.py``.

Usage::

    python tools/compute_sen1floods11_stats.py \
        --data-root data/Sen1Floods11 --modality s1

    python tools/compute_sen1floods11_stats.py \
        --data-root data/Sen1Floods11 --modality s2 --split splits/train.txt
"""
import argparse
import os
import os.path as osp
from typing import List

import numpy as np

try:
    import tifffile
except ImportError as e:
    raise SystemExit(
        'tifffile is required. Install with `pip install tifffile`.') from e


MODALITIES = {
    's1': {
        'subdir': 'S1Hand',
        'suffix': '_S1Hand.tif',
        'channels': 2,
    },
    's2': {
        'subdir': 'S2Hand',
        'suffix': '_S2Hand.tif',
        'channels': 13,
    },
}

LABEL_SUBDIR = 'LabelHand'
LABEL_SUFFIX = '_LabelHand.tif'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True,
                   help='Sen1Floods11 root containing S1Hand/S2Hand/LabelHand')
    p.add_argument('--modality', choices=list(MODALITIES), required=True)
    p.add_argument('--split', default=None,
                   help='Optional txt file listing sample base-names '
                        '(one per line). Paths are resolved against '
                        'data-root.')
    p.add_argument('--max-samples', type=int, default=None,
                   help='Optional cap on the number of images to scan.')
    p.add_argument('--mask-nodata', action='store_true', default=True,
                   help='Exclude pixels where LabelHand == -1 '
                        '(default: True).')
    p.add_argument('--no-mask-nodata', dest='mask_nodata',
                   action='store_false')
    return p.parse_args()


def load_image_list(data_root: str, modality_cfg: dict,
                    split: str) -> List[str]:
    img_dir = osp.join(data_root, modality_cfg['subdir'])
    suffix = modality_cfg['suffix']

    if split:
        split_path = split if osp.isabs(split) else osp.join(data_root, split)
        if not osp.isfile(split_path):
            raise FileNotFoundError(f'split file not found: {split_path}')
        bases = []
        with open(split_path, 'r') as f:
            for line in f:
                base = line.strip()
                if not base:
                    continue
                for s in (suffix, LABEL_SUFFIX):
                    if base.endswith(s):
                        base = base[:-len(s)]
                        break
                bases.append(base)
        img_names = [b + suffix for b in bases]
    else:
        img_names = sorted(f for f in os.listdir(img_dir)
                           if f.endswith(suffix))

    return [osp.join(img_dir, n) for n in img_names]


def label_path_for(img_path: str, modality_cfg: dict, data_root: str) -> str:
    base = osp.basename(img_path)[:-len(modality_cfg['suffix'])]
    return osp.join(data_root, LABEL_SUBDIR, base + LABEL_SUFFIX)


def _hwc(img: np.ndarray) -> np.ndarray:
    """Ensure (H, W, C)."""
    if img.ndim == 2:
        return img[:, :, None]
    if img.ndim == 3 and img.shape[0] < img.shape[-1]:
        return np.transpose(img, (1, 2, 0))
    return img


def compute_stats(img_paths, data_root, modality_cfg, mask_nodata):
    num_channels = modality_cfg['channels']

    # Running sums for per-channel mean / std using Welford-style
    # accumulators (numerically stable and streaming).
    total_count = np.zeros(num_channels, dtype=np.float64)
    total_sum = np.zeros(num_channels, dtype=np.float64)
    total_sqsum = np.zeros(num_channels, dtype=np.float64)

    for i, img_path in enumerate(img_paths):
        img = tifffile.imread(img_path)
        img = _hwc(img).astype(np.float64)
        if img.shape[-1] != num_channels:
            print(f'[warn] skip {img_path}: got {img.shape[-1]} ch '
                  f'(expected {num_channels})')
            continue

        valid_mask = None
        if mask_nodata:
            lbl_path = label_path_for(img_path, modality_cfg, data_root)
            if osp.isfile(lbl_path):
                lbl = tifffile.imread(lbl_path)
                lbl = np.squeeze(lbl)
                valid_mask = (lbl != -1)
            # also drop NaN / Inf pixels that crop up in SAR dB
        finite_mask = np.all(np.isfinite(img), axis=-1)
        if valid_mask is None:
            valid_mask = finite_mask
        else:
            valid_mask &= finite_mask

        if not valid_mask.any():
            continue

        valid = img[valid_mask]           # (N, C)
        total_count += valid.shape[0]
        total_sum += valid.sum(axis=0)
        total_sqsum += (valid ** 2).sum(axis=0)

        if (i + 1) % 50 == 0:
            print(f'  processed {i + 1}/{len(img_paths)} images')

    # Avoid divide-by-zero
    total_count = np.where(total_count == 0, 1, total_count)
    mean = total_sum / total_count
    var = total_sqsum / total_count - mean ** 2
    var = np.clip(var, a_min=0.0, a_max=None)
    std = np.sqrt(var)
    return mean, std, total_count


def main():
    args = parse_args()
    modality_cfg = MODALITIES[args.modality]

    img_paths = load_image_list(args.data_root, modality_cfg, args.split)
    if args.max_samples:
        img_paths = img_paths[:args.max_samples]

    print(f'Found {len(img_paths)} images for modality={args.modality}')
    if not img_paths:
        raise SystemExit('No images matched - check --data-root / --split')

    mean, std, count = compute_stats(
        img_paths, args.data_root, modality_cfg, args.mask_nodata)

    print()
    print('=' * 60)
    print(f'Sen1Floods11 {args.modality} statistics '
          f'(valid pixels: {int(count[0])})')
    print('=' * 60)
    print(f'mean = {mean.tolist()}')
    print(f'std  = {std.tolist()}')
    print()
    print('Paste into NORM_CONFIGS in '
          'mmseg/datasets/transforms/multimodal_pipelines.py:')
    print(f"    '{args.modality}': {{")
    print(f"        'mean': {mean.tolist()},")
    print(f"        'std': {std.tolist()},")
    print('    },')


if __name__ == '__main__':
    main()
