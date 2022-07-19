# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmengine.utils import revert_sync_batchnorm

from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = MODELS.build(cfg.model)
    if not torch.cuda.is_available() or args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    else:
        model = model.to(args.device)
    model.eval()

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
