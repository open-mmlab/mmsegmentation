from collections import OrderedDict


def vit_convert(timm_dict):

    mmseg_dict = OrderedDict()

    for k, v in timm_dict.items():
        if k.startswith('head'):
            continue
        if k.startswith('norm'):
            new_k = k.replace('norm.', 'ln1.')
        elif k.startswith('patch_embed'):
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
        elif k.startswith('blocks'):
            new_k = k.replace('blocks.', 'layers.')
            if 'norm' in new_k:
                new_k = new_k.replace('norm', 'ln')
            elif 'mlp.fc1' in new_k:
                new_k = new_k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in new_k:
                new_k = new_k.replace('mlp.fc2', 'ffn.layers.1')
            elif 'attn.qkv' in new_k:
                new_k = new_k.replace('attn.qkv.', 'attn.attn.in_proj_')
            elif 'attn.proj' in new_k:
                new_k = new_k.replace('attn.proj', 'attn.attn.out_proj')
        else:
            new_k = k
        mmseg_dict[new_k] = v

    return mmseg_dict
