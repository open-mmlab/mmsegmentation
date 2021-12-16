# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_swin(ckpt):
    new_ckpt = OrderedDict()

    ignored_keys = [
        'incre_modules', 'downsamp_modules', 'final_layer', 'classifier'
    ]
    for k, v in ckpt.items():
        if any([k.startswith(kw) for kw in ignored_keys]):
            continue
        elif k.startswith('stage'):
            new_v = v
            if 'mlp.' in k:
                new_k = k.replace('mlp.', 'ffn.')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt[new_k] = new_v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained swin models to'
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_swin(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
