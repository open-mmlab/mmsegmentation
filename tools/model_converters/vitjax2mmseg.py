import argparse
import os.path as osp

import mmcv
import numpy as np
import torch


def vit_jax_to_torch(jax_weights):
    torch_weights = dict()

    # patch embedding
    conv_filters = jax_weights['embedding/kernel']
    # conv_filters = rearrange(conv_filters, 'h w c d -> d c h w')
    conv_filters = torch.permute(conv_filters, (3, 2, 0, 1))
    torch_weights['patch_embed.projection.weight'] = conv_filters
    torch_weights['patch_embed.projection.bias'] = jax_weights[
        'embedding/bias']

    # pos embedding
    torch_weights['pos_embed'] = jax_weights[
        'Transformer/posembed_input/pos_embedding']

    # cls token
    torch_weights['cls_token'] = jax_weights['cls']

    # head
    torch_weights['ln1.weight'] = jax_weights['Transformer/encoder_norm/scale']
    torch_weights['ln1.bias'] = jax_weights['Transformer/encoder_norm/bias']
    # torch_weights["head.weight"] = jax_weights["head/kernel"]
    # torch_weights["head.bias"] = jax_weights["head/bias"]

    # transformer blocks
    for i in range(12):
        jax_block = f'Transformer/encoderblock_{i}'
        torch_block = f'layers.{i}'

        # attention norm
        torch_weights[f'{torch_block}.ln1.weight'] = jax_weights[
            f'{jax_block}/LayerNorm_0/scale']
        torch_weights[f'{torch_block}.ln1.bias'] = jax_weights[
            f'{jax_block}/LayerNorm_0/bias']

        # attention
        query_weight = jax_weights[
            f'{jax_block}/MultiHeadDotProductAttention_1/query/kernel']
        query_bias = jax_weights[
            f'{jax_block}/MultiHeadDotProductAttention_1/query/bias']
        key_weight = jax_weights[
            f'{jax_block}/MultiHeadDotProductAttention_1/key/kernel']
        key_bias = jax_weights[
            f'{jax_block}/MultiHeadDotProductAttention_1/key/bias']
        value_weight = jax_weights[
            f'{jax_block}/MultiHeadDotProductAttention_1/value/kernel']
        value_bias = jax_weights[
            f'{jax_block}/MultiHeadDotProductAttention_1/value/bias']

        qkv_weight = np.stack((query_weight, key_weight, value_weight), 1)
        # qkv_weight = rearrange(qkv_weight, 'out qkv nh d-> out (qkv nh d)')
        qkv_weight = torch.flatten(qkv_weight, start_dim=1)
        qkv_bias = np.stack((query_bias, key_bias, value_bias), 0)
        # qkv_bias = rearrange(qkv_bias, 'qkv nh d -> (qkv nh d)')
        qkv_bias = torch.flatten(qkv_bias, start_dim=0)

        torch_weights[f'{torch_block}.attn.attn.in_proj_weight'] = qkv_weight
        torch_weights[f'{torch_block}.attn.attn.in_proj_bias'] = qkv_bias
        to_out_weight = jax_weights[
            f'{jax_block}/MultiHeadDotProductAttention_1/out/kernel']
        # to_out_weight = rearrange(to_out_weight, 'h hd d -> (h hd) d')
        to_out_weight = torch.flatten(to_out_weight, start_dim=0, end_dim=1)
        torch_weights[
            f'{torch_block}.attn.attn.out_proj.weight'] = to_out_weight
        torch_weights[f'{torch_block}.attn.attn.out_proj.bias'] = jax_weights[
            f'{jax_block}/MultiHeadDotProductAttention_1/out/bias']

        # mlp norm
        torch_weights[f'{torch_block}.ln2.weight'] = jax_weights[
            f'{jax_block}/LayerNorm_2/scale']
        torch_weights[f'{torch_block}.ln2.bias'] = jax_weights[
            f'{jax_block}/LayerNorm_2/bias']

        # mlp
        torch_weights[f'{torch_block}.ffn.layers.0.0.weight'] = jax_weights[
            f'{jax_block}/MlpBlock_3/Dense_0/kernel']
        torch_weights[f'{torch_block}.ffn.layers.0.0.bias'] = jax_weights[
            f'{jax_block}/MlpBlock_3/Dense_0/bias']
        torch_weights[f'{torch_block}.ffn.layers.1.weight'] = jax_weights[
            f'{jax_block}/MlpBlock_3/Dense_1/kernel']
        torch_weights[f'{torch_block}.ffn.layers.1.bias'] = jax_weights[
            f'{jax_block}/MlpBlock_3/Dense_1/bias']

    # transpose weights
    for k, v in torch_weights.items():
        if 'weight' in k and 'patch_embed' not in k and 'ln' not in k:
            # v = rearrange(v, 'i o -> o i')
            v = torch.permute(v, (1, 0))
        torch_weights[k] = torch.tensor(v)

    return torch_weights


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys from jax official pretrained vit models to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    jax_weights = np.load(args.src)
    torch_weights = vit_jax_to_torch(jax_weights)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(torch_weights, args.dst)


if __name__ == '__main__':
    main()
