# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint, wrap_fp16_model
from PIL import Image

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


@torch.no_grad()
def main(args):

    models = []
    gpu_ids = args.gpus
    configs = args.config
    ckpts = args.checkpoint

    cfg = mmcv.Config.fromfile(configs[0])

    if args.aug_test:
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
        ]
        cfg.data.test.pipeline[1].flip = True
    else:
        cfg.data.test.pipeline[1].img_ratios = [1.0]
        cfg.data.test.pipeline[1].flip = False

    torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False,
    )

    for idx, (config, ckpt) in enumerate(zip(configs, ckpts)):
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        if cfg.get('fp16', None):
            wrap_fp16_model(model)
        load_checkpoint(model, ckpt, map_location='cpu')
        torch.cuda.empty_cache()
        tmpdir = args.out
        mmcv.mkdir_or_exist(tmpdir)
        model = MMDataParallel(model, device_ids=[gpu_ids[idx % len(gpu_ids)]])
        model.eval()
        models.append(model)

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    for batch_indices, data in zip(loader_indices, data_loader):
        result = []

        for model in models:
            x, _ = scatter_kwargs(
                inputs=data, kwargs=None, target_gpus=model.device_ids)
            if args.aug_test:
                logits = model.module.aug_test_logits(**x[0])
            else:
                logits = model.module.simple_test_logits(**x[0])
            result.append(logits)

        result_logits = 0
        for logit in result:
            result_logits += logit

        pred = result_logits.argmax(axis=1).squeeze()
        img_info = dataset.img_infos[batch_indices[0]]
        file_name = os.path.join(
            tmpdir, img_info['ann']['seg_map'].split(os.path.sep)[-1])
        Image.fromarray(pred.astype(np.uint8)).save(file_name)
        prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model Ensemble with logits result')
    parser.add_argument(
        '--config', type=str, nargs='+', help='ensemble config files path')
    parser.add_argument(
        '--checkpoint',
        type=str,
        nargs='+',
        help='ensemble checkpoint files path')
    parser.add_argument(
        '--aug-test',
        action='store_true',
        help='control ensemble aug-result or single-result (default)')
    parser.add_argument(
        '--out', type=str, default='results', help='the dir to save result')
    parser.add_argument(
        '--gpus', type=int, nargs='+', default=[0], help='id of gpu to use')

    args = parser.parse_args()
    assert len(args.config) == len(args.checkpoint), \
        f'len(config) must equal len(checkpoint), ' \
        f'but len(config) = {len(args.config)} and' \
        f'len(checkpoint) = {len(args.checkpoint)}'
    assert args.out, "ensemble result out-dir can't be None"
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
