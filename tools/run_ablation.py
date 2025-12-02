#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Automate CM-DGSeg ablation experiments."""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from mmengine.config import Config

ABLATIONS: Dict[str, Dict[str, bool]] = {
    'rgb_only': dict(use_disp=False, enable_dgfe=False, enable_cmg=False,
                     with_boundary=False, geometry_reg=False),
    'early4ch': dict(use_disp=True, enable_dgfe=False, enable_cmg=False,
                     with_boundary=False, geometry_reg=False, early_fusion=True),
    'dual_no_dgfe': dict(use_disp=True, enable_dgfe=False, enable_cmg=False,
                         with_boundary=False, geometry_reg=False),
    '+dgfe': dict(use_disp=True, enable_dgfe=True, enable_cmg=False,
                  with_boundary=False, geometry_reg=False),
    '+cmg': dict(use_disp=True, enable_dgfe=True, enable_cmg=True,
                 with_boundary=False, geometry_reg=False),
    '+edge_loss': dict(use_disp=True, enable_dgfe=True, enable_cmg=True,
                       with_boundary=True, geometry_reg=False),
    '+geo_reg': dict(use_disp=True, enable_dgfe=True, enable_cmg=True,
                     with_boundary=True, geometry_reg=True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run CM-DGSeg ablations')
    parser.add_argument('config', type=str, help='Base CM-DGSeg config file.')
    parser.add_argument('--work-dir', type=str, default='work_dirs/cm_dgseg_ablation',
                        help='Root directory for ablation experiments.')
    parser.add_argument('--iters', type=int, default=None,
                        help='Override training iterations for quick experiments.')
    parser.add_argument('--launcher', type=str, default='none',
                        help='Launcher for distributed training (default: none).')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of GPUs to use.')
    parser.add_argument('--skip-train', action='store_true',
                        help='Only run evaluation assuming checkpoints exist.')
    return parser.parse_args()


def _filter_pipeline(pipeline: List[dict], keep_disp: bool) -> List[dict]:
    filtered = []
    for step in pipeline:
        if not keep_disp and step['type'] in {
                'LoadCityscapesDisparity', 'ConcatRGBDispTo4Ch'}:
            continue
        filtered.append(step)
    return filtered


def _update_dataset_pipeline(cfg: Config, pipeline_key: str, pipeline: List[dict]) -> None:
    cfg[pipeline_key] = pipeline
    dataloader_key = pipeline_key.replace('pipeline', 'dataloader')
    if dataloader_key in cfg and 'dataset' in cfg[dataloader_key]:
        cfg[dataloader_key]['dataset']['pipeline'] = pipeline


def _convert_to_early_fusion(cfg: Config) -> None:
    backbone = deepcopy(cfg.model['backbone_rgb'])
    backbone['in_channels'] = 4
    decode_head = deepcopy(cfg.model['decode_head'])
    cfg.model = dict(
        type='EncoderDecoder',
        data_preprocessor=cfg.model['data_preprocessor'],
        backbone=backbone,
        decode_head=dict(
            type='SegformerHead',
            in_channels=decode_head['in_channels'],
            in_index=decode_head['in_index'],
            channels=decode_head['channels'],
            dropout_ratio=decode_head['dropout_ratio'],
            num_classes=decode_head['num_classes'],
            norm_cfg=decode_head['norm_cfg'],
            align_corners=decode_head['align_corners'],
            loss_decode=decode_head['loss_decode']),
        train_cfg=cfg.get('train_cfg', dict()),
        test_cfg=cfg.get('test_cfg', dict()))
    cfg.model.pop('backbone_disp', None)


def apply_ablation(cfg: Config, tag: str) -> Config:
    options = ABLATIONS[tag]
    cfg = cfg.copy()
    keep_disp = options.get('use_disp', True)

    for pipeline_key in ['train_pipeline', 'test_pipeline', 'tta_pipeline']:
        if pipeline_key in cfg:
            new_pipeline = _filter_pipeline(cfg[pipeline_key], keep_disp)
            _update_dataset_pipeline(cfg, pipeline_key, new_pipeline)

    if options.get('early_fusion'):
        _convert_to_early_fusion(cfg)
    else:
        cfg.model['decode_head']['use_disp'] = options['use_disp']
        cfg.model['decode_head']['enable_dgfe'] = options['enable_dgfe']
        cfg.model['decode_head']['enable_cmg'] = options['enable_cmg']
        cfg.model['decode_head']['with_boundary'] = options['with_boundary']
        if options['with_boundary']:
            cfg.model['decode_head'].setdefault('boundary_loss_weight', 0.3)
        cfg.model['decode_head']['geometry_reg_weight'] = (
            0.05 if options['geometry_reg'] else 0.0)

    if not keep_disp:
        cfg.model.pop('backbone_disp', None)
        dp = cfg.model.get('data_preprocessor', {})
        if 'mean' in dp and len(dp['mean']) > 3:
            dp['mean'] = dp['mean'][:3]
        if 'std' in dp and len(dp['std']) > 3:
            dp['std'] = dp['std'][:3]

    return cfg


def adjust_iterations(cfg: Config, max_iters: int) -> None:
    cfg.train_cfg['max_iters'] = max_iters
    cfg.train_cfg['val_interval'] = max(100, max_iters // 4)
    for sched in cfg.param_scheduler:
        if sched.get('type') == 'PolyLR':
            sched['end'] = max_iters
        if sched.get('end', None) and sched['end'] > max_iters:
            sched['end'] = max_iters


def run_command(cmd: List[str], capture: bool = False) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(cmd, check=True, text=True, capture_output=True)
    return subprocess.run(cmd, check=True)


def parse_metrics(output: str) -> Dict[str, float]:
    metrics = {}
    for match in re.finditer(r'(mIoU|mAcc|aAcc):\s*([0-9.]+)', output):
        metrics[match.group(1)] = float(match.group(2))
    return metrics


def main() -> None:
    args = parse_args()
    base_cfg = Config.fromfile(args.config)
    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    results_path = work_root / 'ablation_results.csv'

    with open(results_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['preset', 'work_dir', 'mIoU'])
        writer.writeheader()
        for preset in ABLATIONS:
            cfg = apply_ablation(base_cfg, preset)
            if args.iters:
                adjust_iterations(cfg, args.iters)
            with tempfile.NamedTemporaryFile('w', suffix=f'_{preset}.py', delete=False) as tmp:
                cfg.dump(tmp.name)
                cfg_path = tmp.name
            preset_work_dir = work_root / preset
            preset_work_dir.mkdir(parents=True, exist_ok=True)

            if not args.skip_train:
                train_cmd = [
                    sys.executable, 'tools/train.py', cfg_path,
                    f'--work-dir={preset_work_dir}',
                    f'--launcher={args.launcher}'
                ]
                if args.launcher != 'none':
                    train_cmd.extend(['--devices', str(args.devices)])
                run_command(train_cmd)

            checkpoint = preset_work_dir / 'latest.pth'
            if not checkpoint.exists():
                raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
            test_cmd = [
                sys.executable, 'tools/test.py', cfg_path, str(checkpoint), '--eval', 'mIoU'
            ]
            test_result = run_command(test_cmd, capture=True)
            metrics = parse_metrics(test_result.stdout)
            writer.writerow({
                'preset': preset,
                'work_dir': str(preset_work_dir),
                'mIoU': metrics.get('mIoU', 0.0)
            })
            print(f'[{preset}] mIoU: {metrics.get("mIoU", 0.0):.4f}')
            os.remove(cfg_path)

    print(f'Ablation results saved to {results_path}')


if __name__ == '__main__':
    main()
