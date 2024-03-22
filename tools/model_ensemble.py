# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import numpy as np
import torch
from PIL import Image
from mmengine import Config, mkdir_or_exist, ProgressBar
from mmengine.runner import load_checkpoint, Runner
from torch.nn.parallel.scatter_gather import scatter_kwargs

from mmseg.models import build_segmentor
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


@torch.no_grad()
def main(args):
    register_all_modules()

    models = []
    configs = args.config
    ckpts = args.checkpoint

    cfg = Config.fromfile(configs[0])

    if args.aug_test:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    torch.backends.cudnn.benchmark = True

    # build the dataloader
    data_loader = Runner.build_dataloader(cfg.test_dataloader)

    for idx, (config, ckpt) in enumerate(zip(configs, ckpts)):
        cfg = Config.fromfile(config)
        cfg.model.pretrained = None

        if args.aug_test:
            cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
            cfg.tta_model.module = cfg.model
            cfg.model = cfg.tta_model

        model = MODELS.build(cfg.model)
        load_checkpoint(model, ckpt, map_location='cuda')
        torch.cuda.empty_cache()
        model.cuda()
        model.eval()
        models.append(model)

    tmpdir = args.out
    mkdir_or_exist(tmpdir)

    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for data in data_loader:
        # print(len(data['inputs']), len(data['data_samples']))
        # print(data['data_samples'][0][0])
        save_name = data['data_samples'][0][0].img_path.split(os.sep)[-1].replace('.jpg', '.png')

        logits_0 = torch.zeros(data['data_samples'][0][0].ori_shape)
        logits_1 = torch.zeros(data['data_samples'][0][0].ori_shape)
        logits_2 = torch.zeros(data['data_samples'][0][0].ori_shape)

        for model in models:
            logits = model.test_step(data)

            logits_0 += logits[0].seg_logits.data.softmax(dim=0).squeeze().detach().cpu()[0, :, :]
            logits_1 += logits[0].seg_logits.data.softmax(dim=0).squeeze().detach().cpu()[1, :, :]
            logits_2 += logits[0].seg_logits.data.softmax(dim=0).squeeze().detach().cpu()[2, :, :]

        logits_0 = logits_0.div(len(models))
        logits_1 = logits_1.div(len(models))
        logits_2 = logits_2.div(len(models))

        final_logits = torch.stack((logits_0, logits_1, logits_2), dim=0)
        # print(final_logits.shape)

        pred = final_logits.argmax(dim=0, keepdim=True).squeeze().detach().cpu().numpy()
        # print(pred.shape)

        save_path = os.path.join(tmpdir, save_name)
        Image.fromarray(pred.astype(np.uint8)).save(save_path)
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
