# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.backbones.clip_ovseg import CLIPOVCATSeg

feature_extractor = dict(
    type='ResNet',
    depth=101,
    num_stages=3,
    out_indices=(0, 1, 2),
    dilations=(1, 1, 1),
    strides=(1, 2, 2),
    norm_eval=False,
    style='pytorch',
    contract_dilation=True)
train_class_json = 'tests/data/coco.json'
test_class_json = 'tests/data/coco.json'
clip_pretrained = 'ViT-B/16'
clip_finetune = 'attention'


def test_clip_ov_catseg():
    """Test CLIPOVCATSeg backbone."""

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Test backbone multiplier
    model = CLIPOVCATSeg(
        feature_extractor,
        train_class_json,
        test_class_json,
        clip_pretrained,
        clip_finetune,
        backbone_multiplier=0).to(device)
    model.train()
    for p in model.feature_extractor.parameters():
        assert not p.requires_grad

    # Test normal inference
    temp = torch.randn((1, 3, 384, 384)).to(device)
    outputs = model(temp)
    assert outputs['appearance_feat'][0].shape == (1, 256, 96, 96)
    assert outputs['appearance_feat'][1].shape == (1, 512, 48, 48)
    assert outputs['appearance_feat'][2].shape == (1, 1024, 24, 24)
    assert outputs['clip_text_feat'].shape == (171, 80, 512)
    assert outputs['clip_text_feat_test'].shape == (171, 80, 512)
    assert outputs['clip_img_feat'].shape == (1, 512, 24, 24)

    # Test finetune CLIP Visual encoder
    model.train()
    for n, p in model.clip_model.visual.named_parameters():
        if clip_finetune in n:
            assert p.requires_grad

    # Test frozen CLIP text encoder
    model.train()
    for p in model.clip_model.transformer.parameters():
        assert not p.requires_grad
