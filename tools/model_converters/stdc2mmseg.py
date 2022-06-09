# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_stdc(ckpt, stdc_type):
    new_state_dict = {}
    if stdc_type == 'STDC1':
        stage_lst = ['0', '1', '2.0', '2.1', '3.0', '3.1', '4.0', '4.1']
    else:
        stage_lst = [
            '0', '1', '2.0', '2.1', '2.2', '2.3', '3.0', '3.1', '3.2', '3.3',
            '3.4', '4.0', '4.1', '4.2'
        ]
    for k, v in ckpt.items():
        ori_k = k
        flag = False
        if 'cp.' in k:
            k = k.replace('cp.', '')
        if 'features.' in k:
            num_layer = int(k.split('.')[1])
            feature_key_lst = 'features.' + str(num_layer) + '.'
            stages_key_lst = 'stages.' + stage_lst[num_layer] + '.'
            k = k.replace(feature_key_lst, stages_key_lst)
            flag = True
        if 'conv_list' in k:
            k = k.replace('conv_list', 'layers')
            flag = True
        if 'avd_layer.' in k:
            if 'avd_layer.0' in k:
                k = k.replace('avd_layer.0', 'downsample.conv')
            elif 'avd_layer.1' in k:
                k = k.replace('avd_layer.1', 'downsample.bn')
            flag = True
        if flag:
            new_state_dict[k] = ckpt[ori_k]

    return new_state_dict


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained STDC1/2 to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    parser.add_argument('type', help='model type: STDC1 or STDC2')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    assert args.type in ['STDC1',
                         'STDC2'], 'STD type should be STDC1 or STDC2!'
    weight = convert_stdc(state_dict, args.type)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
