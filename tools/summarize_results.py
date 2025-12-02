#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Summarize CM-DGSeg evaluation results into CSV, Markdown, and plots."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize evaluation metrics')
    parser.add_argument('paths', nargs='+', help='Evaluation JSON files or directories.')
    parser.add_argument('--output-dir', type=str, default='outputs/summary',
                        help='Directory to store aggregated summaries.')
    parser.add_argument('--metric-key', type=str, default='mIoU',
                        help='Primary metric key to plot (default: mIoU).')
    return parser.parse_args()


def find_result_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            files.extend(sorted(p.rglob('*.json')))
        elif p.suffix == '.json':
            files.append(p)
    return files


def extract_metrics(path: Path) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[-1]
    if 'metrics' in data:
        metrics = data['metrics']
    elif 'metric_values' in data:
        metrics = data['metric_values']
    else:
        metrics = data
    class_iou = {}
    if 'class_iou' in metrics:
        class_iou = {k: float(v) for k, v in metrics['class_iou'].items()}
    elif 'IoUs' in metrics and 'classes' in metrics:
        class_iou = {cls: float(iou) for cls, iou in zip(metrics['classes'], metrics['IoUs'])}
    elif 'per_class_results' in metrics:
        class_iou = {k: float(v.get('IoU', 0.0)) for k, v in metrics['per_class_results'].items()}
    scalars = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    experiment = path.stem
    return experiment, scalars, class_iou


def save_csv(output_dir: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    import csv
    csv_path = output_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_markdown(output_dir: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    md_path = output_dir / 'metrics.md'
    with open(md_path, 'w') as f:
        f.write('| ' + ' | '.join(fieldnames) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(fieldnames)) + ' |\n')
        for row in rows:
            f.write('| ' + ' | '.join(f'{row.get(col, 0):.4f}' if isinstance(row.get(col), float)
                                        else str(row.get(col, ''))
                                        for col in fieldnames) + ' |\n')


def plot_metric(output_dir: Path, rows: List[Dict[str, float]], metric_key: str) -> None:
    plt.figure(figsize=(10, 5))
    names = [row['experiment'] for row in rows]
    values = [row.get(metric_key, 0.0) for row in rows]
    plt.bar(names, values, color='#1f77b4')
    plt.ylabel(metric_key)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f'{metric_key}_bar.png')
    plt.close()


def main() -> None:
    args = parse_args()
    files = find_result_files(args.paths)
    if not files:
        raise FileNotFoundError('No evaluation JSON files found.')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    class_rows: Dict[str, Dict[str, float]] = {}

    for path in files:
        experiment, scalars, class_iou = extract_metrics(path)
        row = {'experiment': experiment}
        row.update(scalars)
        rows.append(row)
        for cls, value in class_iou.items():
            class_rows.setdefault(cls, {})[experiment] = value

    fieldnames = sorted({key for row in rows for key in row.keys()})
    save_csv(output_dir, rows, fieldnames)
    save_markdown(output_dir, rows, fieldnames)
    plot_metric(output_dir, rows, args.metric_key)

    # Save per-class IoU tables
    for cls, values in class_rows.items():
        class_path = output_dir / f'class_{cls}_iou.csv'
        with open(class_path, 'w') as f:
            f.write('experiment,IoU\n')
            for exp, val in values.items():
                f.write(f'{exp},{val:.4f}\n')

    print(f'Summary saved to {output_dir}')


if __name__ == '__main__':
    main()
