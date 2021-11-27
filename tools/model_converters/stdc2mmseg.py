# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_stdc1(ckpt):
    new_state_dict = {}
    for k, v in ckpt.items():
        ori_k = k
        flag = False
        if 'cp.' in k:
            k = k.replace('cp.', '')
        if 'features.0.' in k:
            k = k.replace('features.0.', 'stages.0.')
            flag = True
        if 'features.1.' in k:
            k = k.replace('features.1.', 'stages.1.')
            flag = True
        if 'features.2.' in k:
            k = k.replace('features.2.', 'stages.2.0.')
            flag = True
        if 'features.3.' in k:
            k = k.replace('features.3.', 'stages.2.1.')
            flag = True
        if 'features.4.' in k:
            k = k.replace('features.4.', 'stages.3.0.')
            flag = True
        if 'features.5.' in k:
            k = k.replace('features.5.', 'stages.3.1.')
            flag = True
        if 'features.6.' in k:
            k = k.replace('features.6.', 'stages.4.0.')
            flag = True
        if 'features.7.' in k:
            k = k.replace('features.7.', 'stages.4.1.')
            flag = True
        if 'conv_list' in k:
            k = k.replace('conv_list', 'layers')
            flag = True
        if 'avd_layer.0' in k:
            k = k.replace('avd_layer.0', 'avg_pool.conv')
            flag = True
        if 'avd_layer.1' in k:
            k = k.replace('avd_layer.1', 'avg_pool.bn')
            flag = True
        if 'conv_last' in k:
            k = k.replace('conv_last', 'final_conv')
            flag = True
        if 'arm16' in k:
            k = k.replace('arm16', 'arms.0')
            flag = True
        if 'arm32' in k:
            k = k.replace('arm32', 'arms.1')
            flag = True
        if 'conv_atten' in k:
            k = k.replace('conv_atten', 'conv_attn.conv')
            flag = True
        if 'bn_atten' in k:
            k = k.replace('bn_atten', 'conv_attn.bn')
            flag = True
        if 'conv_head32' in k:
            k = k.replace('conv_head32', 'convs.1')
            flag = True
        if 'conv_head16' in k:
            k = k.replace('conv_head16', 'convs.0')
            flag = True
        if 'ffm.convblk' in k:
            k = k.replace('ffm.convblk', 'ffm.conv0')
            flag = True
        if 'ffm.conv1' in k:
            k = k.replace('ffm.conv1', 'ffm.conv1.conv')
            flag = True
        if 'ffm.conv2' in k:
            k = k.replace('ffm.conv2', 'ffm.conv2.conv')
            flag = True
        if 'conv_out' not in k:
            k = 'backbone.' + k
        if 'conv_out.conv.conv.' in k:
            k = k.replace('conv_out.conv.conv.', 'decode_head.convs.0.conv.')
            flag = True
        if 'conv_out.conv.bn.' in k:
            k = k.replace('conv_out.conv.bn.', 'decode_head.convs.0.bn.')
            flag = True
        if 'conv_out.conv_out' in k:
            k = k.replace('conv_out.conv_out', 'decode_head.conv_seg')
            flag = True
        if 'x' in k:
            flag = False
        if 'conv_avg' in k:
            flag = True
        if flag:
            new_state_dict[k] = ckpt[ori_k]

    return new_state_dict


def convert_stdc2(ckpt):
    new_state_dict = {}
    for k, v in ckpt.items():
        ori_k = k
        flag = False
        if 'cp.' in k:
            k = k.replace('cp.', '')
        if 'features.0.' in k:
            k = k.replace('features.0.', 'stages.0.')
            flag = True
        if 'features.1.' in k:
            k = k.replace('features.1.', 'stages.1.')
            flag = True
        if 'features.2.' in k:
            k = k.replace('features.2.', 'stages.2.0.')
            flag = True
        if 'features.3.' in k:
            k = k.replace('features.3.', 'stages.2.1.')
            flag = True
        if 'features.4.' in k:
            k = k.replace('features.4.', 'stages.2.2.')
            flag = True
        if 'features.5.' in k:
            k = k.replace('features.5.', 'stages.2.3.')
            flag = True
        if 'features.6.' in k:
            k = k.replace('features.6.', 'stages.3.0.')
            flag = True
        if 'features.7.' in k:
            k = k.replace('features.7.', 'stages.3.1.')
            flag = True
        if 'features.8.' in k:
            k = k.replace('features.8.', 'stages.3.2.')
            flag = True
        if 'features.9.' in k:
            k = k.replace('features.9.', 'stages.3.3.')
            flag = True
        if 'features.10.' in k:
            k = k.replace('features.10.', 'stages.3.4.')
            flag = True
        if 'features.11.' in k:
            k = k.replace('features.11.', 'stages.4.0.')
            flag = True
        if 'features.12.' in k:
            k = k.replace('features.12.', 'stages.4.1.')
            flag = True
        if 'features.13.' in k:
            k = k.replace('features.13.', 'stages.4.2.')
            flag = True
        if 'conv_list' in k:
            k = k.replace('conv_list', 'layers')
            flag = True
        if 'avd_layer.0' in k:
            k = k.replace('avd_layer.0', 'avg_pool.conv')
            flag = True
        if 'avd_layer.1' in k:
            k = k.replace('avd_layer.1', 'avg_pool.bn')
            flag = True
        if 'conv_last' in k:
            k = k.replace('conv_last', 'final_conv')
            flag = True
        if 'arm16' in k:
            k = k.replace('arm16', 'arms.0')
            flag = True
        if 'arm32' in k:
            k = k.replace('arm32', 'arms.1')
            flag = True
        if 'conv_atten' in k:
            k = k.replace('conv_atten', 'conv_attn.conv')
            flag = True
        if 'bn_atten' in k:
            k = k.replace('bn_atten', 'conv_attn.bn')
            flag = True
        if 'conv_head32' in k:
            k = k.replace('conv_head32', 'convs.1')
            flag = True
        if 'conv_head16' in k:
            k = k.replace('conv_head16', 'convs.0')
            flag = True
        if 'ffm.convblk' in k:
            k = k.replace('ffm.convblk', 'ffm.conv0')
            flag = True
        if 'ffm.conv1' in k:
            k = k.replace('ffm.conv1', 'ffm.conv1.conv')
            flag = True
        if 'ffm.conv2' in k:
            k = k.replace('ffm.conv2', 'ffm.conv2.conv')
            flag = True
        if 'conv_out' not in k:
            k = 'backbone.' + k
        if 'conv_out.conv.conv.' in k:
            k = k.replace('conv_out.conv.conv.', 'decode_head.convs.0.conv.')
            flag = True
        if 'conv_out.conv.bn.' in k:
            k = k.replace('conv_out.conv.bn.', 'decode_head.convs.0.bn.')
            flag = True
        if 'conv_out.conv_out' in k:
            k = k.replace('conv_out.conv_out', 'decode_head.conv_seg')
            flag = True
        if 'x' in k:
            flag = False
        if 'conv_avg' in k:
            flag = True
        k = k.strip('backbone.')
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

    assert args.type in ['STDC1', 'STDC2'], 'Please check STDC type!'
    if args.type == 'STDC1':
        weight = convert_stdc1(state_dict)
    elif args.type == 'STDC2':
        weight = convert_stdc2(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
