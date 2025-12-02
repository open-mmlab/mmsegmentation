#!/usr/bin/env python3
"""Offline report generator for CM-DGSeg results."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def plot_bar(rows: List[Dict[str, str]], metric: str, out_path: Path, title: str) -> None:
    names = [row.get('experiment', row.get('preset', 'exp')) for row in rows]
    values = [to_float(row.get(metric, 0.0)) for row in rows]
    plt.figure(figsize=(10, 5))
    plt.bar(names, values, color='#2ca02c')
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_param_flops(rows: List[Dict[str, str]], out_path: Path) -> None:
    params = [to_float(row.get('Params(M)', 0.0)) for row in rows]
    flops = [to_float(row.get('FLOPs(G)', 0.0)) for row in rows]
    names = [row.get('experiment', row.get('preset', 'exp')) for row in rows]
    plt.figure(figsize=(10, 5))
    plt.scatter(params, flops, c='#d62728')
    for name, x, y in zip(names, params, flops):
        plt.annotate(name, (x, y))
    plt.xlabel('Params (M)')
    plt.ylabel('FLOPs (G)')
    plt.title('Parameter vs FLOPs trade-off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate CM-DGSeg report figures')
    parser.add_argument('--main-csv', type=str, default='outputs/summary/metrics.csv',
                        help='CSV file containing main results.')
    parser.add_argument('--ablation-csv', type=str, default='work_dirs/cm_dgseg_ablation/ablation_results.csv',
                        help='CSV file containing ablation results.')
    parser.add_argument('--output-dir', type=str, default='paper_figs',
                        help='Directory to store generated figures.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.main_csv).exists():
        main_rows = read_csv(Path(args.main_csv))
        if main_rows:
            plot_bar(main_rows, 'mIoU', output_dir / 'main_mIoU.pdf', 'Main Results mIoU')
            plot_param_flops(main_rows, output_dir / 'params_flops.pdf')
    else:
        print(f'Main CSV not found: {args.main_csv}')

    if Path(args.ablation_csv).exists():
        ablation_rows = read_csv(Path(args.ablation_csv))
        if ablation_rows:
            plot_bar(ablation_rows, 'mIoU', output_dir / 'ablation_mIoU.pdf', 'Ablation mIoU')
    else:
        print(f'Ablation CSV not found: {args.ablation_csv}')

    print(f'Figures saved to {output_dir}')


if __name__ == '__main__':
    main()
