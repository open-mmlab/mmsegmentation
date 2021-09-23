# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp

import pytest
from torch.utils.data import (DistributedSampler, RandomSampler,
                              SequentialSampler)

from mmseg.datasets import (DATASETS, ConcatDataset, build_dataloader,
                            build_dataset)


@DATASETS.register_module()
class ToyDataset(object):

    def __init__(self, cnt=0):
        self.cnt = cnt

    def __item__(self, idx):
        return idx

    def __len__(self):
        return 100


def test_build_dataset():
    cfg = dict(type='ToyDataset')
    dataset = build_dataset(cfg)
    assert isinstance(dataset, ToyDataset)
    assert dataset.cnt == 0
    dataset = build_dataset(cfg, default_args=dict(cnt=1))
    assert isinstance(dataset, ToyDataset)
    assert dataset.cnt == 1

    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_dataset')
    img_dir = 'imgs/'
    ann_dir = 'gts/'

    # We use same dir twice for simplicity
    # with ann_dir
    cfg = dict(
        type='CustomDataset',
        pipeline=[],
        data_root=data_root,
        img_dir=[img_dir, img_dir],
        ann_dir=[ann_dir, ann_dir])
    dataset = build_dataset(cfg)
    assert isinstance(dataset, ConcatDataset)
    assert len(dataset) == 10

    # with ann_dir, split
    cfg = dict(
        type='CustomDataset',
        pipeline=[],
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        split=['splits/train.txt', 'splits/val.txt'])
    dataset = build_dataset(cfg)
    assert isinstance(dataset, ConcatDataset)
    assert len(dataset) == 5

    # with ann_dir, split
    cfg = dict(
        type='CustomDataset',
        pipeline=[],
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=[ann_dir, ann_dir],
        split=['splits/train.txt', 'splits/val.txt'])
    dataset = build_dataset(cfg)
    assert isinstance(dataset, ConcatDataset)
    assert len(dataset) == 5

    # test mode
    cfg = dict(
        type='CustomDataset',
        pipeline=[],
        data_root=data_root,
        img_dir=[img_dir, img_dir],
        test_mode=True,
        classes=('pseudo_class', ))
    dataset = build_dataset(cfg)
    assert isinstance(dataset, ConcatDataset)
    assert len(dataset) == 10

    # test mode with splits
    cfg = dict(
        type='CustomDataset',
        pipeline=[],
        data_root=data_root,
        img_dir=[img_dir, img_dir],
        split=['splits/val.txt', 'splits/val.txt'],
        test_mode=True,
        classes=('pseudo_class', ))
    dataset = build_dataset(cfg)
    assert isinstance(dataset, ConcatDataset)
    assert len(dataset) == 2

    # len(ann_dir) should be zero or len(img_dir) when len(img_dir) > 1
    with pytest.raises(AssertionError):
        cfg = dict(
            type='CustomDataset',
            pipeline=[],
            data_root=data_root,
            img_dir=[img_dir, img_dir],
            ann_dir=[ann_dir, ann_dir, ann_dir])
        build_dataset(cfg)

    # len(splits) should be zero or len(img_dir) when len(img_dir) > 1
    with pytest.raises(AssertionError):
        cfg = dict(
            type='CustomDataset',
            pipeline=[],
            data_root=data_root,
            img_dir=[img_dir, img_dir],
            split=['splits/val.txt', 'splits/val.txt', 'splits/val.txt'])
        build_dataset(cfg)

    # len(splits) == len(ann_dir) when only len(img_dir) == 1 and len(
    # ann_dir) > 1
    with pytest.raises(AssertionError):
        cfg = dict(
            type='CustomDataset',
            pipeline=[],
            data_root=data_root,
            img_dir=img_dir,
            ann_dir=[ann_dir, ann_dir],
            split=['splits/val.txt', 'splits/val.txt', 'splits/val.txt'])
        build_dataset(cfg)


def test_build_dataloader():
    dataset = ToyDataset()
    samples_per_gpu = 3
    # dist=True, shuffle=True, 1GPU
    dataloader = build_dataloader(
        dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=2)
    assert dataloader.batch_size == samples_per_gpu
    assert len(dataloader) == int(math.ceil(len(dataset) / samples_per_gpu))
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert dataloader.sampler.shuffle

    # dist=True, shuffle=False, 1GPU
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=2,
        shuffle=False)
    assert dataloader.batch_size == samples_per_gpu
    assert len(dataloader) == int(math.ceil(len(dataset) / samples_per_gpu))
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert not dataloader.sampler.shuffle

    # dist=True, shuffle=True, 8GPU
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=2,
        num_gpus=8)
    assert dataloader.batch_size == samples_per_gpu
    assert len(dataloader) == int(math.ceil(len(dataset) / samples_per_gpu))
    assert dataloader.num_workers == 2

    # dist=False, shuffle=True, 1GPU
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=2,
        dist=False)
    assert dataloader.batch_size == samples_per_gpu
    assert len(dataloader) == int(math.ceil(len(dataset) / samples_per_gpu))
    assert isinstance(dataloader.sampler, RandomSampler)
    assert dataloader.num_workers == 2

    # dist=False, shuffle=False, 1GPU
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=3,
        workers_per_gpu=2,
        shuffle=False,
        dist=False)
    assert dataloader.batch_size == samples_per_gpu
    assert len(dataloader) == int(math.ceil(len(dataset) / samples_per_gpu))
    assert isinstance(dataloader.sampler, SequentialSampler)
    assert dataloader.num_workers == 2

    # dist=False, shuffle=True, 8GPU
    dataloader = build_dataloader(
        dataset, samples_per_gpu=3, workers_per_gpu=2, num_gpus=8, dist=False)
    assert dataloader.batch_size == samples_per_gpu * 8
    assert len(dataloader) == int(
        math.ceil(len(dataset) / samples_per_gpu / 8))
    assert isinstance(dataloader.sampler, RandomSampler)
    assert dataloader.num_workers == 16
