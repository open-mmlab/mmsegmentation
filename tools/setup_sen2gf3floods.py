"""
Sen2GF3Floods setup: generate train/val/test splits and compute
per-channel normalization statistics for the fused 6-band input
(Sentinel-2 RGBN 4ch + GF-3 HH/HV 2ch).

Expected layout::

    data/Sen2GF3Floods/Sen2GF3Floods/
        sentinel2/<name>.tif    # 4-band (R, G, B, NIR)
        gaofen3/<name>.tif      # 2-band (HH, HV)
        label/<name>.tif        # 1-band (0=bg, 1=flood)

Usage:
    python tools/setup_sen2gf3floods.py --data-root data/Sen2GF3Floods/Sen2GF3Floods
"""

import argparse
import hashlib
import os
import os.path as osp
import sys

import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None

S2_SUBDIR = 'sentinel2'
GF3_SUBDIR = 'gaofen3'
LABEL_SUBDIR = 'label'

S2_CHANNELS = 4
GF3_CHANNELS = 2
TOTAL_CHANNELS = S2_CHANNELS + GF3_CHANNELS


def scan_basenames(data_root: str):
    """Scan label directory for available tiles, return sorted base names."""
    label_dir = osp.join(data_root, LABEL_SUBDIR)
    if not osp.isdir(label_dir):
        print(f'ERROR: label directory not found: {label_dir}', file=sys.stderr)
        sys.exit(1)

    bases = []
    for f in sorted(os.listdir(label_dir)):
        if f.lower().endswith('.tif') or f.lower().endswith('.tiff'):
            bases.append(f)
    return bases


def verify_files(data_root: str, filenames: list):
    """Verify that sentinel2, gaofen3, and label files all exist."""
    missing = []
    for name in filenames:
        for subdir in (S2_SUBDIR, GF3_SUBDIR, LABEL_SUBDIR):
            path = osp.join(data_root, subdir, name)
            if not osp.isfile(path):
                missing.append(path)
    if missing:
        print(f'WARNING: {len(missing)} missing files:')
        for m in missing[:10]:
            print(f'  {m}')
        if len(missing) > 10:
            print(f'  ... and {len(missing) - 10} more')
    return [n for n in filenames
            if all(osp.isfile(osp.join(data_root, sd, n))
                   for sd in (S2_SUBDIR, GF3_SUBDIR, LABEL_SUBDIR))]


def assign_split(name: str, seed: int,
                 train_ratio: float, val_ratio: float) -> str:
    h = hashlib.md5(f'{seed}:{name}'.encode()).hexdigest()
    v = int(h[:8], 16) / 0x1_0000_0000
    if v < train_ratio:
        return 'train'
    elif v < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'


def write_splits(data_root: str, filenames: list,
                 train_ratio: float, val_ratio: float, seed: int):
    splits_dir = osp.join(data_root, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    buckets = {'train': [], 'val': [], 'test': []}
    for name in filenames:
        s = assign_split(name, seed, train_ratio, val_ratio)
        buckets[s].append(name)

    for split, names in buckets.items():
        path = osp.join(splits_dir, f'{split}.txt')
        with open(path, 'w') as f:
            for n in sorted(names):
                f.write(n + '\n')
        print(f'  {split}: {len(names)} tiles -> {path}')

    return buckets


def load_fused_image(data_root: str, name: str) -> np.ndarray:
    """Load sentinel2 (4ch) + gaofen3 (2ch) -> (H, W, 6) float32."""
    s2_path = osp.join(data_root, S2_SUBDIR, name)
    gf3_path = osp.join(data_root, GF3_SUBDIR, name)

    s2 = tifffile.imread(s2_path).astype(np.float32)
    gf3 = tifffile.imread(gf3_path).astype(np.float32)

    # Ensure (H, W, C)
    if s2.ndim == 2:
        s2 = s2[:, :, np.newaxis]
    elif s2.ndim == 3 and s2.shape[0] < s2.shape[-1]:
        s2 = np.transpose(s2, (1, 2, 0))

    if gf3.ndim == 2:
        gf3 = gf3[:, :, np.newaxis]
    elif gf3.ndim == 3 and gf3.shape[0] < gf3.shape[-1]:
        gf3 = np.transpose(gf3, (1, 2, 0))

    # Concatenate along channel axis
    fused = np.concatenate([s2, gf3], axis=-1)
    return fused


def compute_stats(data_root: str, filenames: list):
    """Welford streaming mean/std over 6-band fused images."""
    n = 0
    mean = np.zeros(TOTAL_CHANNELS, dtype=np.float64)
    m2 = np.zeros(TOTAL_CHANNELS, dtype=np.float64)

    for i, name in enumerate(filenames):
        img = load_fused_image(data_root, name)
        h, w, c = img.shape
        assert c == TOTAL_CHANNELS, f'{name}: expected {TOTAL_CHANNELS} ch, got {c}'

        # Load label to mask invalid pixels
        lbl_path = osp.join(data_root, LABEL_SUBDIR, name)
        lbl = tifffile.imread(lbl_path)
        lbl = np.squeeze(lbl).astype(np.int32)

        # Valid: finite pixels with label in {0, 1}
        finite_mask = np.all(np.isfinite(img), axis=-1)
        label_valid = (lbl == 0) | (lbl == 1)
        valid = finite_mask & label_valid

        pixels = img[valid]  # (N, C)
        if pixels.shape[0] == 0:
            continue

        for px in range(pixels.shape[0]):
            n += 1
            delta = pixels[px] - mean
            mean += delta / n
            delta2 = pixels[px] - mean
            m2 += delta * delta2

        if (i + 1) % 50 == 0 or (i + 1) == len(filenames):
            print(f'  stats: {i + 1}/{len(filenames)} tiles, '
                  f'{n:,} valid pixels')

    std = np.sqrt(m2 / max(n - 1, 1))
    return mean, std, n


def main():
    parser = argparse.ArgumentParser(
        description='Setup Sen2GF3Floods: splits + normalization stats')
    parser.add_argument('--data-root', required=True,
                        help='Root dir containing sentinel2/, gaofen3/, label/')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if tifffile is None:
        print('ERROR: pip install tifffile', file=sys.stderr)
        sys.exit(1)

    data_root = args.data_root
    print(f'Data root: {data_root}\n')

    # 1. Scan & verify
    print('Scanning label directory...')
    filenames = scan_basenames(data_root)
    print(f'  Found {len(filenames)} label files.')

    filenames = verify_files(data_root, filenames)
    print(f'  {len(filenames)} tiles with complete s2 + gf3 + label.\n')

    if not filenames:
        print('ERROR: no complete tiles found.', file=sys.stderr)
        sys.exit(1)

    # 2. Splits
    print('Generating splits...')
    buckets = write_splits(data_root, filenames,
                           args.train_ratio, args.val_ratio, args.seed)
    print()

    # 3. Normalization stats (train split only)
    print('Computing normalization stats on train split...')
    mean, std, n_pixels = compute_stats(data_root, buckets['train'])
    print(f'  {n_pixels:,} valid pixels.\n')

    # 4. Print config snippet
    print('=' * 60)
    print('Paste into MultiModalNormalize.NORM_CONFIGS:\n')
    print(f"    'sen2gf3': {{  # {n_pixels:,} valid train pixels")
    print(f"        'mean': {mean.tolist()},")
    print(f"        'std':  {std.tolist()},")
    print('    },')
    print('=' * 60)


if __name__ == '__main__':
    main()
