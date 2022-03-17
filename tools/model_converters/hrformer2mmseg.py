# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def merge_weight(k, v, info):
    if 'k_proj.weight' in k:
        assert info['prefix'] is None and \
               len(info['weight_dict']) == 0
        info['prefix'] = k.rpartition('k_proj')[0]
        info['weight_dict']['k'] = v
    elif 'k_proj.bias' in k:
        cur_prefix = k.rpartition('k_proj')[0]
        assert cur_prefix == info['prefix'] and \
               len(info['bias_dict']) == 0
        info['bias_dict']['k'] = v
    elif 'v_proj.weight' in k:
        cur_prefix = k.rpartition('v_proj')[0]
        assert cur_prefix == info['prefix'] and \
               len(info['weight_dict']) == 1
        info['weight_dict']['v'] = v
    elif 'v_proj.bias' in k:
        cur_prefix = k.rpartition('v_proj')[0]
        assert cur_prefix == info['prefix'] and \
               len(info['bias_dict']) == 1
        info['bias_dict']['v'] = v
    elif 'q_proj.weight' in k:
        cur_prefix = k.rpartition('q_proj')[0]
        assert cur_prefix == info['prefix'] and \
               len(info['weight_dict']) == 2
        info['weight_dict']['q'] = v
    elif 'q_proj.bias' in k:
        cur_prefix = k.rpartition('q_proj')[0]
        assert cur_prefix == info['prefix'] and \
               len(info['bias_dict']) == 2
        info['bias_dict']['q'] = v
        concat_weight = OrderedDict()
        concat_weight[info['prefix'] + 'qkv.weight'] = torch.cat(
            (info['weight_dict']['q'], info['weight_dict']['k'],
             info['weight_dict']['v']),
            dim=0)
        concat_weight[info['prefix'] + 'qkv.bias'] = torch.cat(
            (info['bias_dict']['q'], info['bias_dict']['k'],
             info['bias_dict']['v']),
            dim=0)
        info['prefix'] = None
        info['weight_dict'] = {}
        info['bias_dict'] = {}
        return concat_weight
    else:
        return 0
    return 1


def convert_hrformer(ckpt):
    new_ckpt = OrderedDict()

    ignored_keys = [
        'incre_modules', 'downsamp_modules', 'final_layer', 'classifier'
    ]
    replace_dict = {
        'mlp.fc1.weight': 'ffn.layers.0.weight',
        'mlp.fc1.bias': 'ffn.layers.0.bias',
        'mlp.norm1.weight': 'ffn.layers.1.weight',
        'mlp.norm1.bias': 'ffn.layers.1.bias',
        'mlp.norm1.running_mean': 'ffn.layers.1.running_mean',
        'mlp.norm1.running_var': 'ffn.layers.1.running_var',
        'mlp.norm1.num_batches_tracked': 'ffn.layers.1.num_batches_tracked',
        'mlp.dw3x3.weight': 'ffn.layers.3.weight',
        'mlp.dw3x3.bias': 'ffn.layers.3.bias',
        'mlp.norm2.weight': 'ffn.layers.4.weight',
        'mlp.norm2.bias': 'ffn.layers.4.bias',
        'mlp.norm2.running_mean': 'ffn.layers.4.running_mean',
        'mlp.norm2.running_var': 'ffn.layers.4.running_var',
        'mlp.norm2.num_batches_tracked': 'ffn.layers.4.num_batches_tracked',
        'mlp.fc2.weight': 'ffn.layers.6.weight',
        'mlp.fc2.bias': 'ffn.layers.6.bias',
        'mlp.norm3.weight': 'ffn.layers.7.weight',
        'mlp.norm3.bias': 'ffn.layers.7.bias',
        'mlp.norm3.running_mean': 'ffn.layers.7.running_mean',
        'mlp.norm3.running_var': 'ffn.layers.7.running_var',
        'mlp.norm3.num_batches_tracked': 'ffn.layers.7.num_batches_tracked'
    }
    merge_info = dict(prefix=None, weight_dict={}, bias_dict={})
    for k, v in ckpt.items():
        if any(k.startswith(kw) for kw in ignored_keys):
            continue
        concat_weight = merge_weight(k, v, merge_info)
        if concat_weight == 1:
            continue
        elif concat_weight != 0:
            new_ckpt.update(concat_weight)
            continue

        flag = False
        for pattern, replace_value in replace_dict.items():
            if pattern in k:
                new_k = k.replace(pattern, replace_value)
                new_ckpt[new_k] = v
                flag = True
                break
        if not flag:
            new_ckpt[k] = v

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
    weight = convert_hrformer(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
