# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine import Config
from mmengine.structures import PixelData

from mmseg.models.decode_heads import SideAdapterCLIPHead
from mmseg.structures import SegDataSample
from .utils import list_to_cuda


def test_san_head():
    H, W = (64, 64)
    clip_channels = 64
    img_channels = 4
    num_queries = 40
    out_dims = 64
    num_classes = 19
    cfg = dict(
        num_classes=num_classes,
        deep_supervision_idxs=[4],
        san_cfg=dict(
            in_channels=img_channels,
            embed_dims=128,
            clip_channels=clip_channels,
            num_queries=num_queries,
            cfg_encoder=dict(num_encode_layer=4, mlp_ratio=2, num_heads=2),
            cfg_decoder=dict(
                num_heads=4,
                num_layers=1,
                embed_channels=32,
                mlp_channels=32,
                num_mlp=2,
                rescale=True)),
        maskgen_cfg=dict(
            sos_token_num=num_queries,
            embed_dims=clip_channels,
            out_dims=out_dims,
            num_heads=4,
            mlp_ratio=2),
        train_cfg=dict(
            num_points=100,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='ClassificationCost', weight=2.0),
                    dict(
                        type='CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
                ])),
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_cls_ce',
                loss_weight=2.0,
                class_weight=[1.0] * num_classes + [0.1]),
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_name='loss_mask_ce',
                loss_weight=5.0),
            dict(
                type='DiceLoss',
                ignore_index=None,
                naive_dice=True,
                eps=1,
                loss_name='loss_mask_dice',
                loss_weight=5.0)
        ])

    cfg = Config(cfg)
    head = SideAdapterCLIPHead(**cfg)

    inputs = torch.rand((2, img_channels, H, W))
    clip_feature = [[
        torch.rand((2, clip_channels, H // 2, W // 2)),
        torch.rand((2, clip_channels))
    ],
                    [
                        torch.rand((2, clip_channels, H // 2, W // 2)),
                        torch.rand((2, clip_channels))
                    ],
                    [
                        torch.rand((2, clip_channels, H // 2, W // 2)),
                        torch.rand((2, clip_channels))
                    ],
                    [
                        torch.rand((2, clip_channels, H // 2, W // 2)),
                        torch.rand((2, clip_channels))
                    ]]
    class_embed = torch.rand((num_classes + 1, out_dims))

    data_samples = []
    for i in range(2):
        data_sample = SegDataSample()
        img_meta = {}
        img_meta['img_shape'] = (H, W)
        img_meta['ori_shape'] = (H, W)
        data_sample.gt_sem_seg = PixelData(
            data=torch.randint(0, num_classes, (1, H, W)))
        data_sample.set_metainfo(img_meta)
        data_samples.append(data_sample)

    batch_img_metas = []
    for data_sample in data_samples:
        batch_img_metas.append(data_sample.metainfo)

    if torch.cuda.is_available():
        head = head.cuda()
        data = list_to_cuda([inputs, clip_feature, class_embed])
        for data_sample in data_samples:
            data_sample.gt_sem_seg.data = data_sample.gt_sem_seg.data.cuda()
    else:
        data = [inputs, clip_feature, class_embed]

    # loss test
    loss_dict = head.loss(data, data_samples, None)
    assert isinstance(loss_dict, dict)

    # prediction test
    with torch.no_grad():
        seg_logits = head.predict(data, batch_img_metas, None)
    assert seg_logits.shape == torch.Size((2, num_classes, H, W))
