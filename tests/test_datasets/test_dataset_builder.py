# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmengine.dataset import ConcatDataset, RepeatDataset

from mmseg.datasets import MultiImageMixDataset
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules

register_all_modules()


@DATASETS.register_module()
class ToyDataset:

    def __init__(self, cnt=0):
        self.cnt = cnt

    def __item__(self, idx):
        return idx

    def __len__(self):
        return 100


def test_build_dataset():
    cfg = dict(type='ToyDataset')
    dataset = DATASETS.build(cfg)
    assert isinstance(dataset, ToyDataset)
    assert dataset.cnt == 0
    dataset = DATASETS.build(cfg, default_args=dict(cnt=1))
    assert isinstance(dataset, ToyDataset)
    assert dataset.cnt == 1

    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_dataset')
    data_prefix = dict(img_path='imgs/', seg_map_path='gts/')

    # test RepeatDataset
    cfg = dict(
        type='BaseSegDataset',
        pipeline=[],
        data_root=data_root,
        data_prefix=data_prefix,
        serialize_data=False)
    dataset = DATASETS.build(cfg)
    dataset_repeat = RepeatDataset(dataset=dataset, times=5)
    assert isinstance(dataset_repeat, RepeatDataset)
    assert len(dataset_repeat) == 25

    # test ConcatDataset
    # We use same dir twice for simplicity
    # with data_prefix.seg_map_path
    cfg1 = dict(
        type='BaseSegDataset',
        pipeline=[],
        data_root=data_root,
        data_prefix=data_prefix,
        serialize_data=False)
    cfg2 = dict(
        type='BaseSegDataset',
        pipeline=[],
        data_root=data_root,
        data_prefix=data_prefix,
        serialize_data=False)
    dataset1 = DATASETS.build(cfg1)
    dataset2 = DATASETS.build(cfg2)
    dataset_concat = ConcatDataset(datasets=[dataset1, dataset2])
    assert isinstance(dataset_concat, ConcatDataset)
    assert len(dataset_concat) == 10

    # test MultiImageMixDataset
    dataset = MultiImageMixDataset(dataset=dataset_concat, pipeline=[])
    assert isinstance(dataset, MultiImageMixDataset)
    assert len(dataset) == 10

    cfg = dict(type='ConcatDataset', datasets=[cfg1, cfg2])

    dataset = MultiImageMixDataset(dataset=cfg, pipeline=[])
    assert isinstance(dataset, MultiImageMixDataset)
    assert len(dataset) == 10

    # with data_prefix.seg_map_path, ann_file
    cfg1 = dict(
        type='BaseSegDataset',
        pipeline=[],
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='splits/train.txt',
        serialize_data=False)
    cfg2 = dict(
        type='BaseSegDataset',
        pipeline=[],
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='splits/val.txt',
        serialize_data=False)

    dataset1 = DATASETS.build(cfg1)
    dataset2 = DATASETS.build(cfg2)
    dataset_concat = ConcatDataset(datasets=[dataset1, dataset2])
    assert isinstance(dataset_concat, ConcatDataset)
    assert len(dataset_concat) == 5

    # test mode
    cfg1 = dict(
        type='BaseSegDataset',
        pipeline=[],
        data_root=data_root,
        data_prefix=dict(img_path='imgs/'),
        test_mode=True,
        metainfo=dict(classes=('pseudo_class', )),
        serialize_data=False)
    cfg2 = dict(
        type='BaseSegDataset',
        pipeline=[],
        data_root=data_root,
        data_prefix=dict(img_path='imgs/'),
        test_mode=True,
        metainfo=dict(classes=('pseudo_class', )),
        serialize_data=False)

    dataset1 = DATASETS.build(cfg1)
    dataset2 = DATASETS.build(cfg2)
    dataset_concat = ConcatDataset(datasets=[dataset1, dataset2])
    assert isinstance(dataset_concat, ConcatDataset)
    assert len(dataset_concat) == 10

    # test mode with ann_files
    cfg1 = dict(
        type='BaseSegDataset',
        pipeline=[],
        data_root=data_root,
        data_prefix=dict(img_path='imgs/'),
        ann_file='splits/val.txt',
        test_mode=True,
        metainfo=dict(classes=('pseudo_class', )),
        serialize_data=False)
    cfg2 = dict(
        type='BaseSegDataset',
        pipeline=[],
        data_root=data_root,
        data_prefix=dict(img_path='imgs/'),
        ann_file='splits/val.txt',
        test_mode=True,
        metainfo=dict(classes=('pseudo_class', )),
        serialize_data=False)

    dataset1 = DATASETS.build(cfg1)
    dataset2 = DATASETS.build(cfg2)
    dataset_concat = ConcatDataset(datasets=[dataset1, dataset2])
    assert isinstance(dataset_concat, ConcatDataset)
    assert len(dataset_concat) == 2
