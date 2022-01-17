# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        persistent_workers=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = model.cuda()
    device = next(model.parameters()).device  # model device

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    total_iters = 200

    data_loader_iter = iter(data_loader)
    # benchmark with 200 image and take the average
    for i in range(total_iters):

        data = next(data_loader_iter)

        data = scatter(data, [device])[0]
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {total_iters}], '
                      f'fps: {fps:.2f} img / s')

    fps = (total_iters - num_warmup) / pure_inf_time
    print(f'Overall fps: {fps:.2f} img / s')


if __name__ == '__main__':
    main()
