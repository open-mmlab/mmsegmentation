# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from typing import List

import torch
import torch.nn.functional as F
from huggingface_hub.utils._errors import LocalEntryNotFoundError
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from mmseg.utils import ConfigType
from ..utils import clip_wrapper
from ..utils.clip_templates import (IMAGENET_TEMPLATES,
                                    IMAGENET_TEMPLATES_SELECT)


@MODELS.register_module()
class CLIPOVCATSeg(BaseModule):
    """CLIP based Open Vocabulary CAT-Seg model backbone.

    This backbone is the modified implementation of `CAT-Seg Backbone
    <https://arxiv.org/abs/2303.11797>`_. It combines the CLIP model and
    another feature extractor, a.k.a the appearance guidance extractor
    in the original `CAT-Seg`.

    Args:
        feature_extractor (ConfigType): Appearance guidance extractor
            config dict.
        train_class_json (str): The training class json file.
        test_class_json (str): The path to test class json file.
        clip_pretrained (str): The pre-trained clip type.
        clip_finetune (str): The finetuning settings of clip model.
        custom_clip_weights (str): The custmized clip weights directory. When
            encountering huggingface model download errors, you can manually
            download the pretrained weights.
        backbone_multiplier (float): The learning rate multiplier.
            Default: 0.01.
        prompt_depth (int): The prompt depth. Default: 0.
        prompt_length (int): The prompt length. Default: 0.
        prompt_ensemble_type (str): The prompt ensemble type.
            Default: "imagenet".
        pixel_mean (List[float]): The pixel mean for feature extractor.
        pxiel_std (List[float]): The pixel std for feature extractor.
        clip_pixel_mean (List[float]): The pixel mean for clip model.
        clip_pxiel_std (List[float]): The pixel std for clip model.
        clip_img_feat_size: (List[int]: Clip image embedding size from
            image encoder.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
            self,
            feature_extractor: ConfigType,
            train_class_json: str,
            test_class_json: str,
            clip_pretrained: str,
            clip_finetune: str,
            custom_clip_weights: str = None,
            backbone_multiplier=0.01,
            prompt_depth: int = 0,
            prompt_length: int = 0,
            prompt_ensemble_type: str = 'imagenet',
            pixel_mean: List[float] = [123.675, 116.280, 103.530],
            pixel_std: List[float] = [58.395, 57.120, 57.375],
            clip_pixel_mean: List[float] = [
                122.7709383, 116.7460125, 104.09373615
            ],
            clip_pixel_std: List[float] = [68.5005327, 66.6321579, 70.3231630],
            clip_img_feat_size: List[int] = [24, 24],
            init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # normalization parameters
        self.register_buffer('pixel_mean',
                             torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std',
                             torch.Tensor(pixel_std).view(1, -1, 1, 1), False)
        self.register_buffer('clip_pixel_mean',
                             torch.Tensor(clip_pixel_mean).view(1, -1, 1, 1),
                             False)
        self.register_buffer('clip_pixel_std',
                             torch.Tensor(clip_pixel_std).view(1, -1, 1, 1),
                             False)
        self.clip_resolution = (
            384, 384) if clip_pretrained == 'ViT-B/16' else (336, 336)
        # modified clip image encoder with fixed size dense output
        self.clip_img_feat_size = clip_img_feat_size

        # prepare clip templates
        self.prompt_ensemble_type = prompt_ensemble_type
        if self.prompt_ensemble_type == 'imagenet_select':
            prompt_templates = IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == 'imagenet':
            prompt_templates = IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == 'single':
            prompt_templates = [
                'A photo of a {} in the scene',
            ]
        else:
            raise NotImplementedError
        self.prompt_templates = prompt_templates

        # build the feature extractor
        self.feature_extractor = MODELS.build(feature_extractor)

        # build CLIP model
        with open(train_class_json) as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json) as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts is not None
        if self.test_class_texts is None:
            self.test_class_texts = self.class_texts
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        if clip_pretrained == 'ViT-G' or clip_pretrained == 'ViT-H':
            # for OpenCLIP models
            import open_clip
            name, pretrain = (
                'ViT-H-14',
                'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else (
                    'ViT-bigG-14', 'laion2b_s39b_b160k')
            try:
                open_clip_model = open_clip.create_model_and_transforms(
                    name,
                    pretrained=pretrain,
                    device=device,
                    force_image_size=336,
                )
                clip_model, _, clip_preprocess = open_clip_model
            except ConnectionError or LocalEntryNotFoundError as e:
                print(f'Has {e} when loading weights from huggingface!')
                print(
                    f'Will load {pretrain} weights from {custom_clip_weights}.'
                )
                assert custom_clip_weights is not None, 'Please specify custom weights directory.'  # noqa
                assert os.path.exists(
                    os.path.join(custom_clip_weights,
                                 'open_clip_pytorch_model.bin')
                ), 'Please provide a valid directory for manually downloaded model.'  # noqa
                open_clip_model = open_clip.create_model_and_transforms(
                    name,
                    pretrained=None,
                    device='cpu',
                    force_image_size=336,
                )
                clip_model, _, clip_preprocess = open_clip_model

                open_clip.load_checkpoint(
                    clip_model,
                    os.path.expanduser(
                        os.path.join(custom_clip_weights,
                                     'open_clip_pytorch_model.bin')))
                clip_model.to(torch.device(device))

            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip_wrapper.load(
                clip_pretrained,
                device=device,
                jit=False,
                prompt_depth=prompt_depth,
                prompt_length=prompt_length)

        # pre-encode classes text prompts
        text_features = self.class_embeddings(self.class_texts,
                                              prompt_templates, clip_model,
                                              device).permute(1, 0, 2).float()
        text_features_test = self.class_embeddings(self.test_class_texts,
                                                   prompt_templates,
                                                   clip_model,
                                                   device).permute(1, 0,
                                                                   2).float()
        self.register_buffer('text_features', text_features, False)
        self.register_buffer('text_features_test', text_features_test, False)

        # prepare CLIP model finetune
        self.clip_finetune = clip_finetune
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess

        for name, params in self.clip_model.named_parameters():
            if 'visual' in name:
                if clip_finetune == 'prompt':
                    params.requires_grad = True if 'prompt' in name else False
                elif clip_finetune == 'attention':
                    if 'attn' in name or 'position' in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == 'full':
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        finetune_backbone = backbone_multiplier > 0.
        for name, params in self.feature_extractor.named_parameters():
            if 'norm0' in name:
                params.requires_grad = False
            else:
                params.requires_grad = finetune_backbone

    @torch.no_grad()
    def class_embeddings(self,
                         classnames,
                         templates,
                         clip_model,
                         device='cpu'):
        """Convert class names to text embeddings by clip model.

        Args:
            classnames (list): loaded from json file.
            templates (dict): text template.
            clip_model (nn.Module): prepared clip model.
            device (str | torch.device): loading device of text
                encoder results.
        """
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname)
                         for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).to(device)
            else:
                texts = clip_wrapper.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(
                    len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        return zeroshot_weights

    def custom_normalize(self, inputs):
        """Input normalization for clip model and feature extractor
        respectively.

        Args:
            inputs: batched input images.
        """
        # clip images
        batched_clip = (inputs - self.clip_pixel_mean) / self.clip_pixel_std
        batched_clip = F.interpolate(
            batched_clip,
            size=self.clip_resolution,
            mode='bilinear',
            align_corners=False)
        # feature extractor images
        batched = (inputs - self.pixel_mean) / self.pixel_std
        return batched, batched_clip

    def forward(self, inputs):
        """
        Args:
            inputs: minibatch image. (B, 3, H, W)
        Returns:
            outputs (dict):
            'appearance_feat': list[torch.Tensor], w.r.t. out_indices of
                `self.feature_extractor`.
            'clip_text_feat': the text feature extracted by clip text encoder.
            'clip_text_feat_test': the text feature extracted by clip text
                encoder for testing.
            'clip_img_feat': the image feature extracted clip image encoder.
        """
        inputs, clip_inputs = self.custom_normalize(inputs)
        outputs = dict()
        # extract appearance guidance feature
        outputs['appearance_feat'] = self.feature_extractor(inputs)

        # extract clip features
        outputs['clip_text_feat'] = self.text_features
        outputs['clip_text_feat_test'] = self.text_features_test
        clip_features = self.clip_model.encode_image(
            clip_inputs, dense=True)  # B, 577(24x24+1), C
        B = clip_features.size(0)
        outputs['clip_img_feat'] = clip_features[:, 1:, :].permute(
            0, 2, 1).reshape(B, -1, *self.clip_img_feat_size)

        return outputs
