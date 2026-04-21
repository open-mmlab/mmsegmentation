"""
Overlay flood predictions on real satellite imagery.

For each tile, loads the multi-band satellite TIFF, creates an RGB
composite, and overlays flood pixels (class 1) with a solid color.
Non-flood regions show the real image.

Usage (S2 - true color RGB from bands 4/3/2):
    python tools/overlay_pred_on_image.py \
        --pred-dir work_dirs/generalization/sen1floods11_s2/eval_all/vis_data/vis_image/ \
        --image-dir data/Sen1Floods11/S2Hand/ \
        --label-dir data/Sen1Floods11/LabelHand/ \
        --out-dir work_dirs/generalization/sen1floods11_s2/overlay/ \
        --modal s2

Usage (S1 - false color from VV/VH):
    python tools/overlay_pred_on_image.py \
        --pred-dir work_dirs/generalization/sen1floods11_s1/eval_all/vis_data/vis_image/ \
        --image-dir data/Sen1Floods11/S1Hand/ \
        --label-dir data/Sen1Floods11/LabelHand/ \
        --out-dir work_dirs/generalization/sen1floods11_s1/overlay/ \
        --modal s1
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tifffile
except ImportError:
    tifffile = None

# ── defaults ─────────────────────────────────────────────────────────────────
FLOOD_COLOR = (0, 11, 197)       # #000bc5
FLOOD_RGB_IN_PRED = (255, 0, 0)  # red in palette output

# Sentinel-2 L1C band order in Sen1Floods11 S2Hand TIFFs (13 bands):
#   0:B1  1:B2(Blue)  2:B3(Green)  3:B4(Red)  4:B5  5:B6  6:B7
#   7:B8(NIR)  8:B8A  9:B9  10:B10  11:B11(SWIR1)  12:B12(SWIR2)
BAND_PRESETS = {
    's2':       [3, 2, 1],     # True color (R=B4, G=B3, B=B2)
    's2_false': [7, 3, 2],     # False color NIR (R=B8, G=B4, B=B3)
    's2_swir':  [11, 7, 3],    # SWIR composite
    's1':       [0, 1, 0],     # VV(R), VH(G), VV(B)
}

_MODAL_SUFFIX = {'s1': '_S1Hand.tif', 's2': '_S2Hand.tif'}


def _hex_to_rgb(h: str):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _percentile_stretch(band: np.ndarray,
                        lo: float = 2, hi: float = 98) -> np.ndarray:
    """Stretch a single band to 0-255 using percentile clipping."""
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    vmin, vmax = np.percentile(valid, [lo, hi])
    if vmax - vmin < 1e-6:
        return np.zeros_like(band, dtype=np.uint8)
    stretched = (band - vmin) / (vmax - vmin)
    stretched = np.clip(stretched, 0, 1)
    return (stretched * 255).astype(np.uint8)


def load_rgb_composite(tiff_path: str, bands: list,
                       plo: float = 2, phi: float = 98) -> np.ndarray:
    """Load multi-band TIFF → (H, W, 3) uint8 RGB."""
    data = tifffile.imread(tiff_path).astype(np.float32)  # (C, H, W) or (H, W, C)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    if data.shape[0] > data.shape[-1] and data.ndim == 3:
        # Already (C, H, W)
        pass
    elif data.shape[-1] <= 13 and data.ndim == 3:
        # (H, W, C) → (C, H, W)
        data = data.transpose(2, 0, 1)

    rgb = np.stack([
        _percentile_stretch(np.nan_to_num(data[b], nan=0.0), plo, phi)
        for b in bands
    ], axis=-1)  # (H, W, 3)
    return rgb


def load_pred_classmap(png_path: str) -> np.ndarray:
    """Load prediction PNG → uint8 (H, W): 0=bg, 1=flood."""
    img = np.array(Image.open(png_path).convert('RGB'), dtype=np.uint8)
    flood = (
        (img[:, :, 0] == FLOOD_RGB_IN_PRED[0]) &
        (img[:, :, 1] == FLOOD_RGB_IN_PRED[1]) &
        (img[:, :, 2] == FLOOD_RGB_IN_PRED[2])
    )
    classmap = np.zeros(img.shape[:2], dtype=np.uint8)
    classmap[flood] = 1
    return classmap


def load_nodata_mask(label_path: str) -> np.ndarray:
    """Return bool (H, W): True where GT is nodata (-1)."""
    raw = tifffile.imread(label_path)
    raw = np.squeeze(raw).astype(np.int32)
    return raw == -1


def composite(real_rgb: np.ndarray,
              pred: np.ndarray,
              flood_color: tuple,
              alpha: float = 1.0,
              nodata_mask: np.ndarray = None) -> np.ndarray:
    """Overlay flood predictions on real image.

    - Flood (pred==1): blended with flood_color at given alpha
    - Background (pred==0): real image
    - Nodata: real image (or could dim it)
    """
    out = real_rgb.copy().astype(np.float32)
    flood_mask = pred == 1

    fc = np.array(flood_color, dtype=np.float32)
    out[flood_mask] = (1 - alpha) * out[flood_mask] + alpha * fc

    return np.clip(out, 0, 255).astype(np.uint8)


def find_pred_png(pred_dir: Path, base: str, modal: str) -> Path:
    suffix = _MODAL_SUFFIX[modal].replace('.tif', '')
    pattern = f'{base}{suffix}.tif_*.png'
    matches = sorted(pred_dir.glob(pattern))
    if matches:
        return matches[-1]
    pattern2 = f'{base}{suffix}*.png'
    matches2 = sorted(pred_dir.glob(pattern2))
    return matches2[-1] if matches2 else None


def collect_bases(image_dir: Path, modal: str) -> list:
    suffix = _MODAL_SUFFIX[modal]
    bases = []
    for f in sorted(image_dir.glob(f'*{suffix}')):
        base = f.name[:-len(suffix)]
        bases.append(base)
    return bases


def main():
    parser = argparse.ArgumentParser(
        description='Overlay flood predictions on satellite imagery')
    parser.add_argument('--pred-dir', required=True,
                        help='Dir with prediction PNGs from SegVisualizationHook')
    parser.add_argument('--image-dir', required=True,
                        help='Dir with satellite image TIFFs (S1Hand/ or S2Hand/)')
    parser.add_argument('--label-dir', default=None,
                        help='Dir with LabelHand/ TIFFs (for nodata masking)')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory for overlay PNGs')
    parser.add_argument('--modal', choices=['s1', 's2'], required=True,
                        help='Modality: s1 or s2')
    parser.add_argument('--bands', default=None,
                        help='RGB band indices, e.g. "3,2,1". '
                             'Default: auto from --modal')
    parser.add_argument('--flood-color', default='000bc5',
                        help='Hex color for flood overlay (default: 000bc5)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Opacity of flood overlay (0-1). '
                             '1.0 = solid color, 0.7 = 70%% overlay')
    parser.add_argument('--stretch-lo', type=float, default=2,
                        help='Low percentile for contrast stretch')
    parser.add_argument('--stretch-hi', type=float, default=98,
                        help='High percentile for contrast stretch')
    args = parser.parse_args()

    if tifffile is None:
        print('ERROR: pip install tifffile', file=sys.stderr)
        sys.exit(1)

    pred_dir = Path(args.pred_dir)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    label_dir = Path(args.label_dir) if args.label_dir else None

    if args.bands:
        bands = [int(x) for x in args.bands.split(',')]
    else:
        bands = BAND_PRESETS[args.modal]

    flood_color = _hex_to_rgb(args.flood_color)

    bases = collect_bases(image_dir, args.modal)
    if not bases:
        print(f'ERROR: no {_MODAL_SUFFIX[args.modal]} files in {image_dir}',
              file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Found {len(bases)} tiles.  Bands: {bands}  '
          f'Flood color: #{args.flood_color}  Alpha: {args.alpha}\n')

    done = 0
    skipped = 0

    for base in bases:
        pred_png = find_pred_png(pred_dir, base, args.modal)
        if pred_png is None:
            skipped += 1
            continue

        suffix = _MODAL_SUFFIX[args.modal]
        tiff_path = image_dir / f'{base}{suffix}'
        if not tiff_path.exists():
            skipped += 1
            continue

        rgb = load_rgb_composite(str(tiff_path), bands,
                                 args.stretch_lo, args.stretch_hi)
        pred = load_pred_classmap(str(pred_png))

        if pred.shape != rgb.shape[:2]:
            print(f'  WARN: shape mismatch {base}: '
                  f'pred {pred.shape} vs image {rgb.shape[:2]}',
                  file=sys.stderr)
            skipped += 1
            continue

        nodata = None
        if label_dir:
            lp = label_dir / f'{base}_LabelHand.tif'
            if lp.exists():
                nodata = load_nodata_mask(str(lp))

        result = composite(rgb, pred, flood_color, args.alpha, nodata)

        out_path = out_dir / f'{base}.png'
        Image.fromarray(result).save(out_path)
        done += 1
        print(f'[{done}] {base}.png')

    print(f'\nDone. {done} images saved to {out_dir}/')
    if skipped:
        print(f'Skipped {skipped} tiles (missing pred or image).')


if __name__ == '__main__':
    main()
