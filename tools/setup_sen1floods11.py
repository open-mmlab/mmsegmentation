"""
One-shot Sen1Floods11 setup for fine-tuning.

This script bridges the gap between a freshly-downloaded Sen1Floods11
folder and a training run of
``configs/floodnet/finetune_sen1floods11_{s1,s2}.py``.

Expected on-disk layout::

    data/Sen1Floods11/
        S1Hand/<base>_S1Hand.tif      # 2-band SAR (VV, VH in dB)
        S2Hand/<base>_S2Hand.tif      # 13-band Sentinel-2 MSI
        LabelHand/<base>_LabelHand.tif  # 1-band label, -1=nodata, 0/1 classes

What the script does (all four steps are independent and can be
re-run):

    1. Scan ``<data_root>/LabelHand`` for every available base-name.
    2. Write deterministic train/val/test splits to
       ``<data_root>/splits/{train,val,test}.txt`` based on an MD5 hash
       of the base-name (so the same tile always lands in the same
       split on re-run).
    3. Compute per-channel mean / std for S1Hand (2 ch) and S2Hand
       (13 ch) *using only training-split tiles*, with NaN / Inf pixels
       and label == -1 pixels masked out. This is important because
       the shipped NORM_CONFIGS values in
       ``mmseg/datasets/transforms/multimodal_pipelines.py`` were
       computed on a different split and may not match your copy of
       the data.
    4. Print the NORM_CONFIGS snippet so you can paste it into
       ``multimodal_pipelines.py``.

Usage::

    # full setup (splits + stats for both modalities)
    python tools/setup_sen1floods11.py --data-root data/Sen1Floods11

    # only regenerate splits (e.g. after adding more tiles)
    python tools/setup_sen1floods11.py --data-root data/Sen1Floods11 \\
        --skip-stats

    # only recompute stats (splits already exist)
    python tools/setup_sen1floods11.py --data-root data/Sen1Floods11 \\
        --skip-splits

    # only one modality
    python tools/setup_sen1floods11.py --data-root data/Sen1Floods11 \\
        --modalities s1

The split ratio defaults to 70 / 15 / 15 train / val / test. Pass
``--train-ratio`` / ``--val-ratio`` to change it.
"""
import argparse
import hashlib
import os
import os.path as osp
from typing import List, Tuple

import numpy as np

try:
    import tifffile
except ImportError as e:
    raise SystemExit(
        'tifffile is required. Install with `pip install tifffile`.') from e


# ---------------------------------------------------------------------------
# Layout constants - keep in sync with Sen1Floods11Dataset.MODAL_CONFIG and
# multimodal_pipelines.MultiModalNormalize.NORM_CONFIGS.
# ---------------------------------------------------------------------------
LABEL_SUBDIR = 'LabelHand'
LABEL_SUFFIX = '_LabelHand.tif'

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


def parse_args():
    p = argparse.ArgumentParser(
        description='Sen1Floods11 setup: splits + normalization stats.')
    p.add_argument('--data-root', required=True,
                   help='Sen1Floods11 root with S1Hand/S2Hand/LabelHand.')
    p.add_argument('--train-ratio', type=float, default=0.70,
                   help='Train fraction (default: 0.70).')
    p.add_argument('--val-ratio', type=float, default=0.15,
                   help='Val fraction (default: 0.15). '
                        'Test fraction = 1 - train - val.')
    p.add_argument('--modalities', nargs='+', default=['s1', 's2'],
                   choices=list(MODALITIES),
                   help='Which modalities to compute stats for.')
    p.add_argument('--skip-splits', action='store_true',
                   help='Reuse existing splits/{train,val,test}.txt.')
    p.add_argument('--skip-stats', action='store_true',
                   help='Only generate splits; do not compute stats.')
    p.add_argument('--seed', type=int, default=42,
                   help='Salt prepended to base-names when hashing. '
                        'Change to get a different split assignment.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: scan
# ---------------------------------------------------------------------------
def scan_basenames(data_root: str) -> List[str]:
    label_dir = osp.join(data_root, LABEL_SUBDIR)
    if not osp.isdir(label_dir):
        raise SystemExit(f'LabelHand dir not found: {label_dir}')

    bases = []
    for fn in sorted(os.listdir(label_dir)):
        if fn.endswith(LABEL_SUFFIX):
            bases.append(fn[:-len(LABEL_SUFFIX)])

    if not bases:
        raise SystemExit(
            f'No *{LABEL_SUFFIX} files found in {label_dir}. '
            'Check --data-root.')
    return bases


# ---------------------------------------------------------------------------
# Step 2: splits
# ---------------------------------------------------------------------------
def assign_split(base: str, seed: int,
                 train_ratio: float, val_ratio: float) -> str:
    """Hash-based deterministic split assignment.

    Using a hash (instead of shuffling + slicing) means the assignment
    is stable if the set of base-names grows - a tile that was in
    ``train`` before stays in ``train``.
    """
    h = hashlib.md5(f'{seed}:{base}'.encode('utf-8')).hexdigest()
    # first 8 hex chars -> 32-bit uint in [0, 2^32), normalized to [0, 1)
    v = int(h[:8], 16) / 0x100000000
    if v < train_ratio:
        return 'train'
    if v < train_ratio + val_ratio:
        return 'val'
    return 'test'


def write_splits(data_root: str, bases: List[str],
                 train_ratio: float, val_ratio: float, seed: int
                 ) -> Tuple[List[str], List[str], List[str]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise SystemExit(
            'Invalid split ratios: need train_ratio > 0, val_ratio >= 0, '
            'train + val < 1.')

    splits = {'train': [], 'val': [], 'test': []}
    for base in bases:
        splits[assign_split(base, seed, train_ratio, val_ratio)].append(base)

    splits_dir = osp.join(data_root, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    for name in ('train', 'val', 'test'):
        items = sorted(splits[name])
        out_path = osp.join(splits_dir, f'{name}.txt')
        with open(out_path, 'w') as f:
            if items:
                f.write('\n'.join(items) + '\n')
        print(f'[splits] {name:5s}: {len(items):4d} -> {out_path}')

    total = sum(len(v) for v in splits.values())
    print(f'[splits] total: {total}')
    return splits['train'], splits['val'], splits['test']


def load_existing_splits(data_root: str
                         ) -> Tuple[List[str], List[str], List[str]]:
    def read(name: str) -> List[str]:
        p = osp.join(data_root, 'splits', f'{name}.txt')
        if not osp.isfile(p):
            raise SystemExit(
                f'--skip-splits was set but {p} is missing. '
                'Run without --skip-splits first.')
        with open(p) as f:
            return [ln.strip() for ln in f if ln.strip()]
    return read('train'), read('val'), read('test')


# ---------------------------------------------------------------------------
# Step 3: stats
# ---------------------------------------------------------------------------
def _hwc(img: np.ndarray) -> np.ndarray:
    """Normalize raster shape to (H, W, C)."""
    if img.ndim == 2:
        return img[:, :, None]
    # tifffile typically returns (C, H, W) for multi-band TIFFs -
    # transpose when the leading axis is the smallest.
    if img.ndim == 3 and img.shape[0] < img.shape[-1]:
        return np.transpose(img, (1, 2, 0))
    return img


def compute_stats_for_bases(data_root: str, modality: str,
                            bases: List[str]
                            ) -> Tuple[np.ndarray, np.ndarray, int]:
    modal_cfg = MODALITIES[modality]
    img_dir = osp.join(data_root, modal_cfg['subdir'])
    label_dir = osp.join(data_root, LABEL_SUBDIR)
    num_channels = modal_cfg['channels']

    if not osp.isdir(img_dir):
        raise SystemExit(
            f'[{modality}] image dir not found: {img_dir}')

    # Welford-style streaming accumulators (float64 for precision).
    total_count = 0
    total_sum = np.zeros(num_channels, dtype=np.float64)
    total_sqsum = np.zeros(num_channels, dtype=np.float64)

    n_missing_img = 0
    n_missing_label = 0
    n_channel_mismatch = 0

    for i, base in enumerate(bases):
        img_path = osp.join(img_dir, base + modal_cfg['suffix'])
        lbl_path = osp.join(label_dir, base + LABEL_SUFFIX)

        if not osp.isfile(img_path):
            n_missing_img += 1
            continue

        img = _hwc(tifffile.imread(img_path)).astype(np.float64)
        if img.shape[-1] != num_channels:
            print(f'  [warn] skip {base}: got {img.shape[-1]} channels '
                  f'(expected {num_channels})')
            n_channel_mismatch += 1
            continue

        # Valid = finite everywhere AND not label nodata.
        valid_mask = np.all(np.isfinite(img), axis=-1)
        if osp.isfile(lbl_path):
            lbl = np.squeeze(tifffile.imread(lbl_path))
            if lbl.shape == img.shape[:2]:
                valid_mask &= (lbl != -1)
        else:
            n_missing_label += 1

        if not valid_mask.any():
            continue

        valid = img[valid_mask]           # (N, C)
        total_count += int(valid.shape[0])
        total_sum += valid.sum(axis=0)
        total_sqsum += (valid ** 2).sum(axis=0)

        if (i + 1) % 50 == 0:
            print(f'  [{modality}] processed {i + 1}/{len(bases)} tiles, '
                  f'valid pixels so far: {total_count}')

    if total_count == 0:
        raise SystemExit(
            f'[{modality}] no valid pixels across {len(bases)} tiles '
            f'(missing imgs: {n_missing_img}, '
            f'channel mismatches: {n_channel_mismatch}).')

    mean = total_sum / total_count
    var = np.clip(total_sqsum / total_count - mean ** 2, 0.0, None)
    std = np.sqrt(var)

    print(f'  [{modality}] done. valid pixels={total_count}, '
          f'missing imgs={n_missing_img}, '
          f'missing labels={n_missing_label}, '
          f'channel mismatches={n_channel_mismatch}')
    return mean, std, total_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    data_root = args.data_root.rstrip('/')

    if not osp.isdir(data_root):
        raise SystemExit(f'--data-root not found: {data_root}')

    # Step 1: scan -----------------------------------------------------
    bases = scan_basenames(data_root)
    print(f'Found {len(bases)} tiles under {data_root}/{LABEL_SUBDIR}/')

    # Step 2: splits ---------------------------------------------------
    if args.skip_splits:
        train_bases, val_bases, test_bases = load_existing_splits(data_root)
        print(f'[splits] reusing existing splits: '
              f'{len(train_bases)}/{len(val_bases)}/{len(test_bases)}')
    else:
        train_bases, val_bases, test_bases = write_splits(
            data_root, bases,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    # Step 3 & 4: stats ------------------------------------------------
    if args.skip_stats:
        print('\n[stats] skipped (--skip-stats)')
        return

    all_stats = {}
    for modality in args.modalities:
        print()
        print(f'[stats] computing {modality} mean/std over '
              f'{len(train_bases)} train tiles (NaN/Inf/label==-1 masked)')
        mean, std, count = compute_stats_for_bases(
            data_root, modality, train_bases)
        all_stats[modality] = (mean, std, count)

    print()
    print('=' * 72)
    print('Paste the following into NORM_CONFIGS in')
    print('  mmseg/datasets/transforms/multimodal_pipelines.py')
    print('(replacing the existing \'s1\' / \'s2\' entries)')
    print('=' * 72)
    for modality in args.modalities:
        mean, std, count = all_stats[modality]
        print(f'    \'{modality}\': {{  # {count} valid train pixels')
        print(f'        \'mean\': {mean.tolist()},')
        print(f'        \'std\':  {std.tolist()},')
        print('    },')


if __name__ == '__main__':
    main()
