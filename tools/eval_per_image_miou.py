"""
Per-image mIoU analysis for Sen1Floods11 S1 and S2 predictions.

Loads prediction PNGs (from SegVisualizationHook) and GT label TIFFs,
computes per-image mIoU, groups by region, and selects the best tile
per region where (s1_mIoU + s2_mIoU) / 2 is highest.

Usage:
    python tools/eval_per_image_miou.py \
        --s1-pred-dir work_dirs/generalization/sen1floods11_s1/eval_all/vis_pred_all/vis_data/vis_image/ \
        --s2-pred-dir work_dirs/generalization/sen1floods11_s2/eval_all/vis_pred_all/vis_data/vis_image/ \
        --label-dir data/Sen1Floods11/LabelHand/

    # Or with a split file to restrict which tiles to evaluate:
    python tools/eval_per_image_miou.py \
        --s1-pred-dir ... --s2-pred-dir ... --label-dir ... \
        --split-file data/Sen1Floods11/splits/all.txt
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None

from PIL import Image

REGIONS = [
    'Bolivia', 'Ghana', 'India', 'Mekong', 'Nigeria',
    'Pakistan', 'Paraguay', 'Somalia', 'Spain', 'Sri-Lanka', 'USA',
]

NUM_CLASSES = 2
IGNORE_INDEX = 255

FLOOD_RGB = (255, 0, 0)


def extract_region(base_name: str) -> str:
    """Extract region prefix from a base name like 'Bolivia_23014'."""
    for r in sorted(REGIONS, key=len, reverse=True):
        if base_name.startswith(r + '_'):
            return r
    m = re.match(r'^([A-Za-z-]+)_\d', base_name)
    return m.group(1) if m else 'Unknown'


def load_gt_label(label_path: str) -> np.ndarray:
    """Load GT label TIFF → uint8 (H, W): 0=bg, 1=flood, 255=nodata."""
    raw = tifffile.imread(label_path)
    raw = np.squeeze(raw).astype(np.int32)
    gt = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)
    gt[raw == 0] = 0
    gt[raw == 1] = 1
    return gt


def pred_png_to_classmap(img_path: str) -> np.ndarray:
    """Load a prediction PNG → uint8 (H, W): 0=bg, 1=flood."""
    img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    flood_mask = (
        (img[:, :, 0] == FLOOD_RGB[0]) &
        (img[:, :, 1] == FLOOD_RGB[1]) &
        (img[:, :, 2] == FLOOD_RGB[2])
    )
    classmap = np.zeros(img.shape[:2], dtype=np.uint8)
    classmap[flood_mask] = 1
    return classmap


def compute_miou(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute per-class IoU and mIoU, ignoring pixels where gt==255."""
    valid = gt != IGNORE_INDEX
    pred_v = pred[valid]
    gt_v = gt[valid]

    ious = {}
    for c in range(NUM_CLASSES):
        inter = np.sum((pred_v == c) & (gt_v == c))
        union = np.sum((pred_v == c) | (gt_v == c))
        if union == 0:
            ious[c] = float('nan')
        else:
            ious[c] = inter / union

    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    miou = np.mean(valid_ious) if valid_ious else float('nan')
    return {'per_class': ious, 'mIoU': miou}


def find_pred_png(pred_dir: Path, base_name: str,
                  modal_suffix: str) -> Path:
    """Find the prediction PNG for a given base name.

    Prediction filenames look like:
      <base>_<ModalSuffix>.tif_<step>.png
    e.g. Bolivia_23014_S1Hand.tif_0.png
    """
    pattern = f'{base_name}_{modal_suffix}.tif_*.png'
    matches = sorted(pred_dir.glob(pattern))
    if matches:
        return matches[-1]
    pattern2 = f'{base_name}_{modal_suffix}*.png'
    matches2 = sorted(pred_dir.glob(pattern2))
    if matches2:
        return matches2[-1]
    return None


def collect_base_names(label_dir: Path, split_file: str = None) -> list:
    """Collect all base names (e.g. 'Bolivia_23014') from labels or split."""
    if split_file and os.path.isfile(split_file):
        bases = []
        with open(split_file) as f:
            for line in f:
                b = line.strip()
                if not b:
                    continue
                b = re.sub(r'_(S1Hand|S2Hand|LabelHand)\.tif$', '', b,
                           flags=re.IGNORECASE)
                bases.append(b)
        return sorted(set(bases))

    bases = []
    for f in sorted(label_dir.glob('*_LabelHand.tif')):
        base = f.name.replace('_LabelHand.tif', '')
        bases.append(base)
    return bases


def main():
    parser = argparse.ArgumentParser(
        description='Per-image mIoU analysis for Sen1Floods11 S1 & S2')
    parser.add_argument('--s1-pred-dir', required=True,
                        help='Dir with S1 prediction PNGs')
    parser.add_argument('--s2-pred-dir', required=True,
                        help='Dir with S2 prediction PNGs')
    parser.add_argument('--label-dir', required=True,
                        help='Path to LabelHand/ directory')
    parser.add_argument('--split-file', default=None,
                        help='Optional split file (one base per line)')
    parser.add_argument('--csv', default=None,
                        help='Optional CSV output path')
    args = parser.parse_args()

    if tifffile is None:
        print('ERROR: pip install tifffile', file=sys.stderr)
        sys.exit(1)

    s1_dir = Path(args.s1_pred_dir)
    s2_dir = Path(args.s2_pred_dir)
    label_dir = Path(args.label_dir)

    for d, name in [(s1_dir, '--s1-pred-dir'),
                    (s2_dir, '--s2-pred-dir'),
                    (label_dir, '--label-dir')]:
        if not d.is_dir():
            print(f'ERROR: {name} "{d}" is not a directory', file=sys.stderr)
            sys.exit(1)

    bases = collect_base_names(label_dir, args.split_file)
    if not bases:
        print('ERROR: no samples found', file=sys.stderr)
        sys.exit(1)
    print(f'Found {len(bases)} tiles.\n')

    rows = []
    skipped = 0

    for base in bases:
        label_path = label_dir / f'{base}_LabelHand.tif'
        if not label_path.exists():
            skipped += 1
            continue

        s1_png = find_pred_png(s1_dir, base, 'S1Hand')
        s2_png = find_pred_png(s2_dir, base, 'S2Hand')
        if s1_png is None or s2_png is None:
            skipped += 1
            continue

        gt = load_gt_label(str(label_path))
        s1_pred = pred_png_to_classmap(str(s1_png))
        s2_pred = pred_png_to_classmap(str(s2_png))

        if s1_pred.shape != gt.shape:
            print(f'  WARN: shape mismatch {base} S1 '
                  f'{s1_pred.shape} vs GT {gt.shape}', file=sys.stderr)
            skipped += 1
            continue
        if s2_pred.shape != gt.shape:
            print(f'  WARN: shape mismatch {base} S2 '
                  f'{s2_pred.shape} vs GT {gt.shape}', file=sys.stderr)
            skipped += 1
            continue

        s1_res = compute_miou(s1_pred, gt)
        s2_res = compute_miou(s2_pred, gt)

        s1_miou = s1_res['mIoU']
        s2_miou = s2_res['mIoU']

        both_valid = (not np.isnan(s1_miou)) and (not np.isnan(s2_miou))
        avg_miou = (s1_miou + s2_miou) / 2 if both_valid else float('nan')

        region = extract_region(base)
        rows.append(dict(
            base=base, region=region,
            s1_mIoU=s1_miou, s2_mIoU=s2_miou,
            s1_bg_iou=s1_res['per_class'][0],
            s1_flood_iou=s1_res['per_class'][1],
            s2_bg_iou=s2_res['per_class'][0],
            s2_flood_iou=s2_res['per_class'][1],
            avg_mIoU=avg_miou,
        ))

    if skipped:
        print(f'Skipped {skipped} tiles (missing pred or label).\n')

    if not rows:
        print('ERROR: no valid tiles evaluated.', file=sys.stderr)
        sys.exit(1)

    # ── Per-image table ──────────────────────────────────────────────
    header = (f'{"Tile":<35} {"Region":<12} '
              f'{"S1_mIoU":>8} {"S2_mIoU":>8} {"Avg":>8}')
    sep = '-' * len(header)
    print(header)
    print(sep)
    for r in sorted(rows, key=lambda x: (x['region'], x['base'])):
        s1 = f"{r['s1_mIoU']:.4f}" if not np.isnan(r['s1_mIoU']) else '  N/A '
        s2 = f"{r['s2_mIoU']:.4f}" if not np.isnan(r['s2_mIoU']) else '  N/A '
        av = f"{r['avg_mIoU']:.4f}" if not np.isnan(r['avg_mIoU']) else '  N/A '
        print(f'{r["base"]:<35} {r["region"]:<12} {s1:>8} {s2:>8} {av:>8}')

    # ── Per-region summary ───────────────────────────────────────────
    region_rows = defaultdict(list)
    for r in rows:
        region_rows[r['region']].append(r)

    print(f'\n{"="*70}')
    print(f'{"Region":<12} {"#Tiles":>6} '
          f'{"S1_mean":>8} {"S2_mean":>8} {"Avg_mean":>8}  '
          f'Best tile (by avg mIoU)')
    print('-' * 90)

    best_per_region = {}
    for region in sorted(region_rows.keys()):
        rr = region_rows[region]
        s1_vals = [x['s1_mIoU'] for x in rr if not np.isnan(x['s1_mIoU'])]
        s2_vals = [x['s2_mIoU'] for x in rr if not np.isnan(x['s2_mIoU'])]
        avg_vals = [x['avg_mIoU'] for x in rr if not np.isnan(x['avg_mIoU'])]

        s1_mean = np.mean(s1_vals) if s1_vals else float('nan')
        s2_mean = np.mean(s2_vals) if s2_vals else float('nan')
        avg_mean = np.mean(avg_vals) if avg_vals else float('nan')

        valid_rr = [x for x in rr if not np.isnan(x['avg_mIoU'])]
        if valid_rr:
            best = max(valid_rr, key=lambda x: x['avg_mIoU'])
        else:
            best = rr[0] if rr else None

        best_per_region[region] = best

        s1s = f'{s1_mean:.4f}' if not np.isnan(s1_mean) else '  N/A '
        s2s = f'{s2_mean:.4f}' if not np.isnan(s2_mean) else '  N/A '
        avs = f'{avg_mean:.4f}' if not np.isnan(avg_mean) else '  N/A '
        best_name = best['base'] if best else 'N/A'
        best_avg = f"{best['avg_mIoU']:.4f}" if best and not np.isnan(
            best['avg_mIoU']) else 'N/A'
        print(f'{region:<12} {len(rr):>6} '
              f'{s1s:>8} {s2s:>8} {avs:>8}  '
              f'{best_name} ({best_avg})')

    # ── Best tile per region (final selection) ───────────────────────
    print(f'\n{"="*70}')
    print('Best tile per region (highest avg of S1 + S2 mIoU):')
    print(f'{"Region":<12} {"Tile":<35} '
          f'{"S1_mIoU":>8} {"S2_mIoU":>8} {"Avg":>8}')
    print('-' * 80)
    for region in sorted(best_per_region.keys()):
        b = best_per_region[region]
        if b is None:
            continue
        s1 = f"{b['s1_mIoU']:.4f}" if not np.isnan(b['s1_mIoU']) else '  N/A '
        s2 = f"{b['s2_mIoU']:.4f}" if not np.isnan(b['s2_mIoU']) else '  N/A '
        av = f"{b['avg_mIoU']:.4f}" if not np.isnan(b['avg_mIoU']) else '  N/A '
        print(f'{region:<12} {b["base"]:<35} {s1:>8} {s2:>8} {av:>8}')

    # ── Optional CSV ─────────────────────────────────────────────────
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write('base,region,s1_mIoU,s2_mIoU,avg_mIoU,'
                    's1_bg_iou,s1_flood_iou,s2_bg_iou,s2_flood_iou\n')
            for r in sorted(rows, key=lambda x: (x['region'], x['base'])):
                f.write(f"{r['base']},{r['region']},"
                        f"{r['s1_mIoU']:.6f},{r['s2_mIoU']:.6f},"
                        f"{r['avg_mIoU']:.6f},"
                        f"{r['s1_bg_iou']:.6f},{r['s1_flood_iou']:.6f},"
                        f"{r['s2_bg_iou']:.6f},{r['s2_flood_iou']:.6f}\n")
        print(f'\nCSV saved to {csv_path}')

    # ── Global summary ───────────────────────────────────────────────
    all_s1 = [r['s1_mIoU'] for r in rows if not np.isnan(r['s1_mIoU'])]
    all_s2 = [r['s2_mIoU'] for r in rows if not np.isnan(r['s2_mIoU'])]
    print(f'\nGlobal S1 mIoU: {np.mean(all_s1):.4f} '
          f'(over {len(all_s1)} tiles)')
    print(f'Global S2 mIoU: {np.mean(all_s2):.4f} '
          f'(over {len(all_s2)} tiles)')


if __name__ == '__main__':
    main()
