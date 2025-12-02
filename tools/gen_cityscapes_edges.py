#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Generate thin binary edge maps from Cityscapes annotations."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate Cityscapes edge maps')
    parser.add_argument('--ann-root', type=str, required=True,
                        help='Root directory to Cityscapes gtFine annotations.')
    parser.add_argument('--out-root', type=str, required=True,
                        help='Output directory for generated edge maps.')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help='Dataset splits to process (default: train val).')
    parser.add_argument('--low', type=float, default=5.0,
                        help='Low threshold for Canny edge detector.')
    parser.add_argument('--high', type=float, default=15.0,
                        help='High threshold for Canny edge detector.')
    parser.add_argument('--kernel', type=int, default=3,
                        help='Kernel size for morphological refinement.')
    return parser.parse_args()


def build_edge_map(label: np.ndarray, low: float, high: float,
                   kernel_size: int) -> np.ndarray:
    if label.ndim == 3:
        label = label[..., 0]
    label_u8 = label.astype(np.uint8)
    canny = cv2.Canny(label_u8, low, high)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gradient = cv2.morphologyEx(label_u8, cv2.MORPH_GRADIENT, kernel)
    combined = np.clip(canny.astype(np.uint16) + gradient.astype(np.uint16), 0,
                       255)
    refined = cv2.dilate(combined.astype(np.uint8), kernel, iterations=1)
    edge = (refined > 0).astype(np.uint8) * 255
    return edge


def main() -> None:
    args = parse_args()
    ann_root = Path(args.ann_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_dir = ann_root / split
        if not split_dir.exists():
            print(f'[Skip] Split {split} not found in {ann_root}.')
            continue
        files = sorted(split_dir.rglob('*_labelIds.png'))
        if not files:
            print(f'[Skip] No label files under {split_dir}.')
            continue
        for ann_path in tqdm(files, desc=f'Processing {split}'):
            rel_path = ann_path.relative_to(ann_root)
            out_path = out_root / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            label = cv2.imread(str(ann_path), cv2.IMREAD_UNCHANGED)
            edge = build_edge_map(label, args.low, args.high, args.kernel)
            cv2.imwrite(str(out_path), edge)

    print(f'Edge maps saved to: {out_root}')


if __name__ == '__main__':
    main()
