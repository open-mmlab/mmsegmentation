# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_vitlayer(paras):
    new_para_name = ''
    if paras[0] == 'ln_1':
        new_para_name = '.'.join(['ln1'] + paras[1:])
    elif paras[0] == 'attn':
        new_para_name = '.'.join(['attn.attn'] + paras[1:])
    elif paras[0] == 'ln_2':
        new_para_name = '.'.join(['ln2'] + paras[1:])
    elif paras[0] == 'mlp':
        if paras[1] == 'c_fc':
            new_para_name = '.'.join(['ffn.layers.0.0'] + paras[-1:])
        else:
            new_para_name = '.'.join(['ffn.layers.1'] + paras[-1:])
    else:
        print(f'Wrong for {paras}')
    return new_para_name


def convert_translayer(paras):
    new_para_name = ''
    if paras[0] == 'attn':
        new_para_name = '.'.join(['attentions.0.attn'] + paras[1:])
    elif paras[0] == 'ln_1':
        new_para_name = '.'.join(['norms.0'] + paras[1:])
    elif paras[0] == 'ln_2':
        new_para_name = '.'.join(['norms.1'] + paras[1:])
    elif paras[0] == 'mlp':
        if paras[1] == 'c_fc':
            new_para_name = '.'.join(['ffns.0.layers.0.0'] + paras[2:])
        elif paras[1] == 'c_proj':
            new_para_name = '.'.join(['ffns.0.layers.1'] + paras[2:])
        else:
            print(f'Wrong for {paras}')
    else:
        print(f'Wrong for {paras}')
    return new_para_name


def convert_key_name(ckpt, visual_split):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        key_list = k.split('.')
        if key_list[0] == 'visual':
            new_transform_name = 'image_encoder'
            if key_list[1] == 'class_embedding':
                new_name = '.'.join([new_transform_name, 'cls_token'])
            elif key_list[1] == 'positional_embedding':
                new_name = '.'.join([new_transform_name, 'pos_embed'])
            elif key_list[1] == 'conv1':
                new_name = '.'.join([
                    new_transform_name, 'patch_embed.projection', key_list[2]
                ])
            elif key_list[1] == 'ln_pre':
                new_name = '.'.join(
                    [new_transform_name, key_list[1], key_list[2]])
            elif key_list[1] == 'transformer':
                new_layer_name = 'layers'
                layer_index = key_list[3]
                paras = key_list[4:]
                if int(layer_index) < visual_split:
                    new_para_name = convert_vitlayer(paras)
                    new_name = '.'.join([
                        new_transform_name, new_layer_name, layer_index,
                        new_para_name
                    ])
                else:
                    new_para_name = convert_translayer(paras)
                    new_transform_name = 'decode_head.rec_with_attnbias'
                    new_layer_name = 'layers'
                    layer_index = str(int(layer_index) - visual_split)
                    new_name = '.'.join([
                        new_transform_name, new_layer_name, layer_index,
                        new_para_name
                    ])
            elif key_list[1] == 'proj':
                new_name = 'decode_head.rec_with_attnbias.proj.weight'
            elif key_list[1] == 'ln_post':
                new_name = k.replace('visual', 'decode_head.rec_with_attnbias')
            else:
                print(f'pop parameter: {k}')
                continue
        else:
            text_encoder_name = 'text_encoder'
            if key_list[0] == 'transformer':
                layer_name = 'transformer'
                layer_index = key_list[2]
                paras = key_list[3:]
                new_para_name = convert_translayer(paras)
                new_name = '.'.join([
                    text_encoder_name, layer_name, layer_index, new_para_name
                ])
            elif key_list[0] in [
                    'positional_embedding', 'text_projection', 'bg_embed',
                    'attn_mask', 'logit_scale', 'token_embedding', 'ln_final'
            ]:
                new_name = 'text_encoder.' + k
            else:
                print(f'pop parameter: {k}')
                continue
        new_ckpt[new_name] = v

    return new_ckpt


def convert_tensor(ckpt):
    cls_token = ckpt['image_encoder.cls_token']
    new_cls_token = cls_token.unsqueeze(0).unsqueeze(0)
    ckpt['image_encoder.cls_token'] = new_cls_token
    pos_embed = ckpt['image_encoder.pos_embed']
    new_pos_embed = pos_embed.unsqueeze(0)
    ckpt['image_encoder.pos_embed'] = new_pos_embed
    proj_weight = ckpt['decode_head.rec_with_attnbias.proj.weight']
    new_proj_weight = proj_weight.transpose(1, 0)
    ckpt['decode_head.rec_with_attnbias.proj.weight'] = new_proj_weight
    return ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    if any([s in args.src for s in ['B-16', 'b16', 'base_patch16']]):
        visual_split = 9
    elif any([s in args.src for s in ['L-14', 'l14', 'large_patch14']]):
        visual_split = 18
    else:
        print('Make sure the clip model is ViT-B/16 or ViT-L/14!')
        visual_split = -1
    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if isinstance(checkpoint, torch.jit.RecursiveScriptModule):
        state_dict = checkpoint.state_dict()
    else:
        if 'state_dict' in checkpoint:
            # timm checkpoint
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            # deit checkpoint
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    weight = convert_key_name(state_dict, visual_split)
    weight = convert_tensor(weight)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
