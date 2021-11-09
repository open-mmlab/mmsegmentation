# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_vit(ckpt):

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('backbone.blocks'):
            if 'norm1' in k:
                new_k = k.replace('norm1', 'ln1')
            elif 'norm2' in k:
                new_k = k.replace('norm2', 'ln2')
            elif 'attn.proj.' in k
                new_k = k.replace('proj.', 'attn.out_proj.')
            else:
                new_k = k
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    state_dict = checkpoint
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_vit(state_dict)
    checkpoint['state_dict'] = weight
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(checkpoint, args.dst)


if __name__ == '__main__':
    main()
