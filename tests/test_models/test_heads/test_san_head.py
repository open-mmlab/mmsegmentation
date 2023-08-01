# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine import Config
from mmengine.structures import PixelData

from mmseg.models.decode_heads import SideAdapterCLIPHead
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList
from .utils import list_to_cuda


def test_san_head():
    H, W = (64, 64)
    clip_channels = 768
    img_channels = 4
    num_queries = 100
    out_dims = 512
    num_classes = 19
    cfg = dict(
        num_classes=num_classes,
        san_cfg=dict(
            in_channels=img_channels,
            embed_dims=240,
            clip_channels=clip_channels,
            num_queries=num_queries,
            cfg_encoder=dict(num_encode_layer=12, mlp_ratio=4, num_heads=12),
            cfg_decoder=dict(
                num_heads=12,
                num_layers=1,
                embed_channels=256,
                mlp_channels=256,
                num_mlp=3,
                rescale=True)),
        maskgen_cfg=dict(
            sos_token_num=num_queries,
            embed_dims=clip_channels,
            out_dims=out_dims))
    cfg = Config(cfg)

    head = SideAdapterCLIPHead(**cfg)

    inputs = torch.rand((1, img_channels, H, W))
    clip_feature = [[
        torch.rand((1, clip_channels, H // 2, W // 2)),
        torch.rand((1, clip_channels))
    ],
                    [
                        torch.rand((1, clip_channels, H // 2, W // 2)),
                        torch.rand((1, clip_channels))
                    ],
                    [
                        torch.rand((1, clip_channels, H // 2, W // 2)),
                        torch.rand((1, clip_channels))
                    ],
                    [
                        torch.rand((1, clip_channels, H // 2, W // 2)),
                        torch.rand((1, clip_channels))
                    ]]
    class_embed = torch.rand((num_classes + 1, out_dims))

    data_samples: SampleList = []
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
    else:
        data = [inputs, clip_feature, class_embed]

    with torch.no_grad():
        mask_props, mask_logits = head.forward(data)
    assert len(mask_props) == len(mask_logits)
    for mask_prop, mask_logit in zip(mask_props, mask_logits):
        assert mask_prop.shape == (1, num_queries, H // 16, W // 16)
        assert mask_logit.shape == (1, num_queries, num_classes + 1)

    with torch.no_grad():
        seg_logits = head.predict(data, batch_img_metas, None)
    assert seg_logits.shape == torch.Size((1, num_classes, H, W))
