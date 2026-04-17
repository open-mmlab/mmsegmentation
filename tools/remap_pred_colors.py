"""
Remap prediction image colors and mask nodata regions.

Converts the two-color palette output from SegVisualizationHook:
  - Black  [  0,   0,   0] (Background) -> #7c7c7c [124, 124, 124]
  - Red    [255,   0,   0] (Flood)      -> #000bc5 [  0,  11, 197]

When --label-dir is given, pixels whose GT label is nodata (-1) are
painted with --nodata-color (default: white #ffffff) so they are
visually distinct from Background.

Usage:
    # Basic remap (no nodata handling):
    python tools/remap_pred_colors.py --src vis_pred/vis_data/vis_image/

    # With nodata masking from GT labels:
    python tools/remap_pred_colors.py \
        --src vis_pred/vis_data/vis_image/ \
        --label-dir data/Sen1Floods11/LabelHand/ \
        --nodata-color ffffff

    # Save to a new directory:
    python tools/remap_pred_colors.py \
        --src vis_pred/vis_data/vis_image/ \
        --dst vis_pred/vis_data/vis_image_remapped/ \
        --label-dir data/Sen1Floods11/LabelHand/
"""

import argparse
import os.path as osp
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tifffile
except ImportError:
    tifffile = None

COLOR_MAP = {
    (0,   0,   0):   (124, 124, 124),   # Background  black  -> #7c7c7c
    (255, 0,   0):   (0,   11,  197),   # Flood       red    -> #000bc5
}

_MODAL_SUFFIX_RE = re.compile(r'_(S1Hand|S2Hand)\.tif', re.IGNORECASE)


def _hex_to_rgb(h: str):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _pred_name_to_label_path(pred_name: str, label_dir: Path) -> Path:
    """Derive the LabelHand TIFF path from a prediction PNG filename.

    Prediction names look like ``Bolivia_23014_S1Hand.tif_0.png``.
    The corresponding label is ``Bolivia_23014_LabelHand.tif``.
    """
    stem = pred_name
    # Strip trailing ``_<step>.png``
    stem = re.sub(r'_\d+\.png$', '', stem)
    # Strip ``.tif`` if still attached
    stem = re.sub(r'\.tif$', '', stem, flags=re.IGNORECASE)
    # Replace modal suffix with LabelHand
    stem = _MODAL_SUFFIX_RE.sub('', stem)
    return label_dir / f'{stem}_LabelHand.tif'


def _load_nodata_mask(label_path: Path) -> np.ndarray:
    """Return bool mask (H, W): True where GT is nodata."""
    if tifffile is None:
        raise ImportError('tifffile is required for --label-dir')
    raw = tifffile.imread(str(label_path))
    raw = np.squeeze(raw)
    return raw.astype(np.int32) == -1


def remap_image(img_array: np.ndarray,
                nodata_mask: np.ndarray = None,
                nodata_rgb: tuple = (255, 255, 255)) -> np.ndarray:
    out = img_array.copy()
    for src_rgb, dst_rgb in COLOR_MAP.items():
        mask = (
            (img_array[:, :, 0] == src_rgb[0]) &
            (img_array[:, :, 1] == src_rgb[1]) &
            (img_array[:, :, 2] == src_rgb[2])
        )
        out[mask] = dst_rgb
    if nodata_mask is not None and nodata_mask.any():
        out[nodata_mask] = nodata_rgb
    return out


def process_file(src_path: Path, dst_path: Path,
                 label_dir: Path = None,
                 nodata_rgb: tuple = (255, 255, 255)) -> None:
    img = Image.open(src_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)

    nodata_mask = None
    if label_dir is not None:
        label_path = _pred_name_to_label_path(src_path.name, label_dir)
        if label_path.exists():
            nd = _load_nodata_mask(label_path)
            if nd.shape == arr.shape[:2]:
                nodata_mask = nd
            else:
                print(f'  WARN: shape mismatch {nd.shape} vs '
                      f'{arr.shape[:2]}, skipping nodata mask for '
                      f'{src_path.name}', file=sys.stderr)
        else:
            print(f'  WARN: label not found: {label_path}',
                  file=sys.stderr)

    arr_out = remap_image(arr, nodata_mask, nodata_rgb)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_out).save(dst_path)


def collect_png(root: Path):
    return sorted(root.rglob('*.png'))


def main():
    parser = argparse.ArgumentParser(
        description='Remap prediction colors and optionally mask nodata')
    parser.add_argument('--src', required=True,
                        help='Source directory (or single PNG).')
    parser.add_argument('--dst', default=None,
                        help='Destination directory. Default: in-place.')
    parser.add_argument('--label-dir', default=None,
                        help='Path to LabelHand/ directory with GT TIFFs. '
                             'When given, nodata pixels (label=-1) are '
                             'painted with --nodata-color.')
    parser.add_argument('--nodata-color', default='ffffff',
                        help='Hex color for nodata pixels (default: ffffff).')
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f'ERROR: --src "{src}" does not exist.', file=sys.stderr)
        sys.exit(1)

    label_dir = Path(args.label_dir) if args.label_dir else None
    if label_dir and not label_dir.is_dir():
        print(f'ERROR: --label-dir "{label_dir}" is not a directory.',
              file=sys.stderr)
        sys.exit(1)

    nodata_rgb = _hex_to_rgb(args.nodata_color)

    if src.is_file():
        pairs = [(src, Path(args.dst) if args.dst else src)]
    else:
        files = collect_png(src)
        if not files:
            print(f'No PNG files found under "{src}".', file=sys.stderr)
            sys.exit(0)
        if args.dst is None:
            pairs = [(f, f) for f in files]
        else:
            dst_root = Path(args.dst)
            pairs = [(f, dst_root / f.relative_to(src)) for f in files]

    for i, (src_f, dst_f) in enumerate(pairs, 1):
        process_file(src_f, dst_f, label_dir, nodata_rgb)
        print(f'[{i}/{len(pairs)}] {src_f}  ->  {dst_f}')

    print(f'\nDone. {len(pairs)} image(s) remapped.')


if __name__ == '__main__':
    main()
