#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize CM-DGSeg attention and gating maps."""

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config

from mmseg.apis import init_model
from mmseg.datasets.pipelines.load_disparity import LoadCityscapesDisparity
from mmseg.registry import DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize CM-DGSeg outputs')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file.')
    parser.add_argument('--output-dir', type=str, default='outputs/vis',
                        help='Directory to save visualization results.')
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 2],
                        help='Dataset indices to visualize.')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='val',
                        help='Dataset split to sample from.')
    return parser.parse_args()


def colorize_mask(mask: np.ndarray, palette: List[List[int]]) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, value in enumerate(palette):
        if idx >= len(palette):
            break
        color[mask == idx] = value
    return color


def tensor_to_heatmap(tensor: torch.Tensor, target_size: tuple) -> np.ndarray:
    tensor = tensor.float().mean(dim=1, keepdim=True)
    tensor = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
    tensor = tensor.sigmoid()
    array = tensor.squeeze().cpu().numpy()
    array = (array - array.min()) / (array.max() - array.min() + 1e-6)
    array = (array * 255).astype(np.uint8)
    return array


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_model(cfg, args.checkpoint, device=device)
    model.eval()

    dataset_cfg = cfg.val_dataloader if args.split == 'val' else cfg.train_dataloader
    dataset = DATASETS.build(dataset_cfg['dataset'])
    palette = dataset.PALETTE if hasattr(dataset, 'PALETTE') else dataset.METAINFO['palette']

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    disp_loader = LoadCityscapesDisparity()

    for idx in args.indices:
        sample = dataset[idx]
        inputs = sample['inputs']
        data_sample = sample['data_samples']
        batch = model.data_preprocessor(dict(inputs=[inputs], data_samples=[data_sample]),
                                        training=False)
        with torch.no_grad():
            seg_logits = model._forward(batch['inputs'], batch['data_samples'])
            results = model.postprocess_result(seg_logits, batch['data_samples'])
            latest = model.decode_head.get_latest_results()

        pred_sample = results[0]
        pred_mask = pred_sample.pred_sem_seg.data.squeeze().cpu().numpy()
        gt_mask = pred_sample.gt_sem_seg.data.squeeze().cpu().numpy()
        height, width = pred_mask.shape
        target_size = (height, width)

        img_path = pred_sample.metainfo.get('img_path', None)
        data_root = dataset.data_root if hasattr(dataset, 'data_root') else ''
        if img_path is None:
            img_path = data_sample.metainfo.get('img_path')
        if img_path is None:
            raise RuntimeError('Image path metadata is missing.')
        if not os.path.isabs(img_path):
            img_path = os.path.join(data_root, img_path)
        rgb = mmcv.imread(img_path)

        disp_dict = {'img_path': img_path}
        disp_info = disp_loader.transform(disp_dict)
        disp_norm = disp_info['disp']
        disp_img = (disp_norm * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_img, cv2.COLORMAP_TURBO)

        rgb_out = rgb.copy()
        pred_color = colorize_mask(pred_mask, palette)
        gt_color = colorize_mask(gt_mask, palette)

        edge_maps = latest.get('a_edge', [])
        conf_maps = latest.get('a_conf', [])
        gate_maps = latest.get('gate', [])
        edge_heat = tensor_to_heatmap(edge_maps[0], target_size) if edge_maps else None
        base_feat = edge_maps[0] if edge_maps else (gate_maps[0] if gate_maps else None)
        if conf_maps:
            conf_tensor = conf_maps[0].mean(dim=1, keepdim=True)
            if base_feat is not None:
                conf_tensor = conf_tensor.expand(-1, 1, base_feat.shape[2], base_feat.shape[3])
            conf_heat = tensor_to_heatmap(conf_tensor, target_size)
        else:
            conf_heat = None
        gate_heat = tensor_to_heatmap(gate_maps[0], target_size) if gate_maps else None

        stem = Path(img_path).stem
        mmcv.imwrite(rgb_out, out_dir / f'{stem}_rgb.png')
        mmcv.imwrite(disp_color, out_dir / f'{stem}_disp.png')
        mmcv.imwrite(pred_color[:, :, ::-1], out_dir / f'{stem}_pred.png')
        mmcv.imwrite(gt_color[:, :, ::-1], out_dir / f'{stem}_gt.png')
        if edge_heat is not None:
            mmcv.imwrite(edge_heat, out_dir / f'{stem}_a_edge.png')
        if conf_heat is not None:
            mmcv.imwrite(conf_heat, out_dir / f'{stem}_a_conf.png')
        if gate_heat is not None:
            mmcv.imwrite(gate_heat, out_dir / f'{stem}_gate.png')

    print(f'Visualizations saved to {out_dir}')


if __name__ == '__main__':
    main()
