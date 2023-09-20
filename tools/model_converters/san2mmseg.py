# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_key_name(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        key_list = k.split('.')
        if key_list[0] == 'clip_visual_extractor':
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
            elif key_list[1] == 'resblocks':
                new_layer_name = 'layers'
                layer_index = key_list[2]
                paras = key_list[3:]
                if paras[0] == 'ln_1':
                    new_para_name = '.'.join(['ln1'] + key_list[4:])
                elif paras[0] == 'attn':
                    new_para_name = '.'.join(['attn.attn'] + key_list[4:])
                elif paras[0] == 'ln_2':
                    new_para_name = '.'.join(['ln2'] + key_list[4:])
                elif paras[0] == 'mlp':
                    if paras[1] == 'c_fc':
                        new_para_name = '.'.join(['ffn.layers.0.0'] +
                                                 key_list[-1:])
                    else:
                        new_para_name = '.'.join(['ffn.layers.1'] +
                                                 key_list[-1:])
                new_name = '.'.join([
                    new_transform_name, new_layer_name, layer_index,
                    new_para_name
                ])
        elif key_list[0] == 'side_adapter_network':
            decode_head_name = 'decode_head'
            module_name = 'side_adapter_network'
            if key_list[1] == 'vit_model':
                if key_list[2] == 'blocks':
                    layer_name = 'encode_layers'
                    layer_index = key_list[3]
                    paras = key_list[4:]
                    if paras[0] == 'norm1':
                        new_para_name = '.'.join(['ln1'] + key_list[5:])
                    elif paras[0] == 'attn':
                        new_para_name = '.'.join(key_list[4:])
                        new_para_name = new_para_name.replace(
                            'attn.qkv.', 'attn.attn.in_proj_')
                        new_para_name = new_para_name.replace(
                            'attn.proj', 'attn.attn.out_proj')
                    elif paras[0] == 'norm2':
                        new_para_name = '.'.join(['ln2'] + key_list[5:])
                    elif paras[0] == 'mlp':
                        new_para_name = '.'.join(['ffn'] + key_list[5:])
                        new_para_name = new_para_name.replace(
                            'fc1', 'layers.0.0')
                        new_para_name = new_para_name.replace(
                            'fc2', 'layers.1')
                    else:
                        print(f'Wrong for {k}')
                    new_name = '.'.join([
                        decode_head_name, module_name, layer_name, layer_index,
                        new_para_name
                    ])
                elif key_list[2] == 'pos_embed':
                    new_name = '.'.join(
                        [decode_head_name, module_name, 'pos_embed'])
                elif key_list[2] == 'patch_embed':
                    new_name = '.'.join([
                        decode_head_name, module_name, 'patch_embed',
                        'projection', key_list[4]
                    ])
                else:
                    print(f'Wrong for {k}')
            elif key_list[1] == 'query_embed' or key_list[
                    1] == 'query_pos_embed':
                new_name = '.'.join(
                    [decode_head_name, module_name, key_list[1]])
            elif key_list[1] == 'fusion_layers':
                layer_name = 'conv_clips'
                layer_index = key_list[2][-1]
                paras = '.'.join(key_list[3:])
                new_para_name = paras.replace('input_proj.0', '0')
                new_para_name = new_para_name.replace('input_proj.1', '1.conv')
                new_name = '.'.join([
                    decode_head_name, module_name, layer_name, layer_index,
                    new_para_name
                ])
            elif key_list[1] == 'mask_decoder':
                new_name = 'decode_head.' + k
            else:
                print(f'Wrong for {k}')
        elif key_list[0] == 'clip_rec_head':
            module_name = 'rec_with_attnbias'
            if key_list[1] == 'proj':
                new_name = '.'.join(
                    [decode_head_name, module_name, 'proj.weight'])
            elif key_list[1] == 'ln_post':
                new_name = '.'.join(
                    [decode_head_name, module_name, 'ln_post', key_list[2]])
            elif key_list[1] == 'resblocks':
                new_layer_name = 'layers'
                layer_index = key_list[2]
                paras = key_list[3:]
                if paras[0] == 'ln_1':
                    new_para_name = '.'.join(['norms.0'] + paras[1:])
                elif paras[0] == 'attn':
                    new_para_name = '.'.join(['attentions.0.attn'] + paras[1:])
                elif paras[0] == 'ln_2':
                    new_para_name = '.'.join(['norms.1'] + paras[1:])
                elif paras[0] == 'mlp':
                    if paras[1] == 'c_fc':
                        new_para_name = '.'.join(['ffns.0.layers.0.0'] +
                                                 paras[2:])
                    elif paras[1] == 'c_proj':
                        new_para_name = '.'.join(['ffns.0.layers.1'] +
                                                 paras[2:])
                    else:
                        print(f'Wrong for {k}')
                new_name = '.'.join([
                    decode_head_name, module_name, new_layer_name, layer_index,
                    new_para_name
                ])
            else:
                print(f'Wrong for {k}')
        elif key_list[0] == 'ov_classifier':
            text_encoder_name = 'text_encoder'
            if key_list[1] == 'transformer':
                layer_name = 'transformer'
                layer_index = key_list[3]
                paras = key_list[4:]
                if paras[0] == 'attn':
                    new_para_name = '.'.join(['attentions.0.attn'] + paras[1:])
                elif paras[0] == 'ln_1':
                    new_para_name = '.'.join(['norms.0'] + paras[1:])
                elif paras[0] == 'ln_2':
                    new_para_name = '.'.join(['norms.1'] + paras[1:])
                elif paras[0] == 'mlp':
                    if paras[1] == 'c_fc':
                        new_para_name = '.'.join(['ffns.0.layers.0.0'] +
                                                 paras[2:])
                    elif paras[1] == 'c_proj':
                        new_para_name = '.'.join(['ffns.0.layers.1'] +
                                                 paras[2:])
                    else:
                        print(f'Wrong for {k}')
                else:
                    print(f'Wrong for {k}')
                new_name = '.'.join([
                    text_encoder_name, layer_name, layer_index, new_para_name
                ])
            elif key_list[1] in [
                    'positional_embedding', 'text_projection', 'bg_embed',
                    'attn_mask', 'logit_scale', 'token_embedding', 'ln_final'
            ]:
                new_name = k.replace('ov_classifier', 'text_encoder')
            else:
                print(f'Wrong for {k}')
        elif key_list[0] == 'criterion':
            new_name = k
        else:
            print(f'Wrong for {k}')
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

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_key_name(state_dict)
    weight = convert_tensor(weight)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
