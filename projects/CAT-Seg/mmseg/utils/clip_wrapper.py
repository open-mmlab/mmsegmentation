# Copyright (c) OpenMMLab. All rights reserved.
# Referred to: https://github.com/KU-CVLAB/CAT-Seg/blob/main/cat_seg/third_party/clip.py # noqa
import hashlib
import os
import urllib
import warnings
from typing import List, Union

import torch
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from tqdm import tqdm

from .clip_model import build_model
from .tokenizer import SimpleTokenizer as _Tokenizer

__all__ = ['available_models', 'load', 'tokenize']
_tokenizer = _Tokenizer()

_MODELS = {
    'RN50':
    'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt',  # noqa
    'RN101':
    'https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt',  # noqa
    'RN50x4':
    'https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt',  # noqa
    'RN50x16':
    'https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt',  # noqa
    'RN50x64':
    'https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt',  # noqa
    'ViT-B/32':
    'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',  # noqa
    'ViT-B/16':
    'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt',  # noqa
    'ViT-L/14':
    'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt',  # noqa
    'ViT-L/14@336px':
    'https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt',  # noqa
}


def _download(url: str, root: str = os.path.expanduser('~/.cache/clip')):
    """Download clip pretrained weights."""
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split('/')[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f'{download_target} exists and is not a regular file')

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target,
                               'rb').read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f'{download_target} exists, but the SHA256 checksum does not\
                match; re-downloading the file')

    with urllib.request.urlopen(url) as source, open(download_target,
                                                     'wb') as output:
        with tqdm(
                total=int(source.info().get('Content-Length')),
                ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target,
                           'rb').read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            'Model has been downloaded but the SHA256 checksum does not not\
                match')

    return download_target


def available_models():
    """Returns a list of available models."""
    return list(_MODELS.keys())


def load(name: str,
         device: Union[str, torch.device] = 'cuda'
         if torch.cuda.is_available() else 'cpu',
         jit=True,
         prompt_depth=0,
         prompt_length=0):
    """Load target clip model."""
    if name not in _MODELS:
        raise RuntimeError(
            f'Model {name} not found; available models = {available_models()}')

    model_path = _download(_MODELS[name])
    model = torch.jit.load(
        model_path, map_location=device if jit else 'cpu').eval()
    n_px = model.input_resolution.item()

    transform = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert('RGB'),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    if not jit:
        model = build_model(model.state_dict(), prompt_depth,
                            prompt_length).to(device)
        return model, transform

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [
        n for n in device_holder.graph.findAllNodes('prim::Constant')
        if 'Device' in repr(n)
    ][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, 'graph') else []
        if hasattr(module, 'forward1'):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes('prim::Constant'):
                if 'value' in node.attributeNames() and str(
                        node['value']).startswith('cuda'):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if device == 'cpu':
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode('aten::to').inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, 'graph') else []
            if hasattr(module, 'forward1'):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes('aten::to'):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        # dtype can be the second or third argument to
                        # aten::to()
                        if inputs[i].node()['value'] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, transform


def load_custom(name: str,
                device: Union[str, torch.device] = 'cuda'
                if torch.cuda.is_available() else 'cpu',
                jit=True,
                n_px=224):
    """Load a customized clip model."""
    if name not in _MODELS:
        raise RuntimeError(
            f'Model {name} not found; available models = {available_models()}')

    model_path = _download(_MODELS[name])
    model = torch.jit.load(
        model_path, map_location=device if jit else 'cpu').eval()
    # n_px = model.input_resolution.item()

    transform = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert('RGB'),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    if not jit:
        model = build_model(model.state_dict()).to(device)
        return model, transform

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [
        n for n in device_holder.graph.findAllNodes('prim::Constant')
        if 'Device' in repr(n)
    ][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, 'graph') else []
        if hasattr(module, 'forward1'):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes('prim::Constant'):
                if 'value' in node.attributeNames() and str(
                        node['value']).startswith('cuda'):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if device == 'cpu':
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode('aten::to').inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, 'graph') else []
            if hasattr(module, 'forward1'):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes('aten::to'):
                    inputs = list(node.inputs())
                    for i in [
                            1, 2
                    ]:  # dtype can be the second or third argument to
                        # aten::to()
                        if inputs[i].node()['value'] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, transform


def tokenize(texts: Union[str, List[str]], context_length: int = 77):
    """Convert texts to tokens."""
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder['<|startoftext|>']
    eot_token = _tokenizer.encoder['<|endoftext|>']
    # encode each template text phrase
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(
                f'Input {texts[i]} is too long for context length\
                    {context_length}')
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
