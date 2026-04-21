"""
Remap prediction image colors to RGBA.

Converts the two-color palette output from SegVisualizationHook:
  - Black  [  0,   0,   0] (Background / Nodata) -> fully transparent
  - Red    [255,   0,   0] (Flood)               -> #000bc5 [0, 11, 197] opaque

Output is RGBA PNG so flood regions can be overlaid on any base map.

Usage:
    # In-place:
    python tools/remap_pred_colors.py --src vis_pred/vis_data/vis_image/

    # Save to a new directory:
    python tools/remap_pred_colors.py \
        --src vis_pred/vis_data/vis_image/ \
        --dst vis_pred/vis_data/vis_image_remapped/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FLOOD_SRC = (255, 0, 0)
FLOOD_DST = (0, 11, 197)


def remap_image(img_array: np.ndarray) -> np.ndarray:
    """Convert RGB prediction to RGBA: flood → #000bc5 opaque, rest → transparent."""
    h, w = img_array.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA, default fully transparent

    flood_mask = (
        (img_array[:, :, 0] == FLOOD_SRC[0]) &
        (img_array[:, :, 1] == FLOOD_SRC[1]) &
        (img_array[:, :, 2] == FLOOD_SRC[2])
    )
    out[flood_mask, 0] = FLOOD_DST[0]
    out[flood_mask, 1] = FLOOD_DST[1]
    out[flood_mask, 2] = FLOOD_DST[2]
    out[flood_mask, 3] = 255  # fully opaque

    return out


def process_file(src_path: Path, dst_path: Path) -> None:
    img = Image.open(src_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    arr_out = remap_image(arr)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_out, 'RGBA').save(dst_path)


def collect_png(root: Path):
    return sorted(root.rglob('*.png'))


def main():
    parser = argparse.ArgumentParser(
        description='Remap prediction image colors '
                    '(black→#7c7c7c, red→#000bc5)')
    parser.add_argument('--src', required=True,
                        help='Source directory (or single PNG).')
    parser.add_argument('--dst', default=None,
                        help='Destination directory. Default: in-place.')
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f'ERROR: --src "{src}" does not exist.', file=sys.stderr)
        sys.exit(1)

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
        process_file(src_f, dst_f)
        print(f'[{i}/{len(pairs)}] {src_f}  ->  {dst_f}')

    print(f'\nDone. {len(pairs)} image(s) remapped.')


if __name__ == '__main__':
    main()
