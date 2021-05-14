import os.path as osp
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmseg.core.evaluation import get_classes, get_palette
from mmseg.datasets import (DATASETS, ADE20KDataset, CityscapesDataset,
                            ConcatDataset, CustomDataset, PascalVOCDataset,
                            RepeatDataset)


def test_classes():
    assert list(CityscapesDataset.CLASSES) == get_classes('cityscapes')
    assert list(PascalVOCDataset.CLASSES) == get_classes('voc') == get_classes(
        'pascal_voc')
    assert list(
        ADE20KDataset.CLASSES) == get_classes('ade') == get_classes('ade20k')

    with pytest.raises(ValueError):
        get_classes('unsupported')


def test_palette():
    assert CityscapesDataset.PALETTE == get_palette('cityscapes')
    assert PascalVOCDataset.PALETTE == get_palette('voc') == get_palette(
        'pascal_voc')
    assert ADE20KDataset.PALETTE == get_palette('ade') == get_palette('ade20k')

    with pytest.raises(ValueError):
        get_palette('unsupported')


@patch('mmseg.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmseg.datasets.CustomDataset.__getitem__',
       MagicMock(side_effect=lambda idx: idx))
def test_dataset_wrapper():
    # CustomDataset.load_annotations = MagicMock()
    # CustomDataset.__getitem__ = MagicMock(side_effect=lambda idx: idx)
    dataset_a = CustomDataset(img_dir=MagicMock(), pipeline=[])
    len_a = 10
    dataset_a.img_infos = MagicMock()
    dataset_a.img_infos.__len__.return_value = len_a
    dataset_b = CustomDataset(img_dir=MagicMock(), pipeline=[])
    len_b = 20
    dataset_b.img_infos = MagicMock()
    dataset_b.img_infos.__len__.return_value = len_b

    concat_dataset = ConcatDataset([dataset_a, dataset_b])
    assert concat_dataset[5] == 5
    assert concat_dataset[25] == 15
    assert len(concat_dataset) == len(dataset_a) + len(dataset_b)

    repeat_dataset = RepeatDataset(dataset_a, 10)
    assert repeat_dataset[5] == 5
    assert repeat_dataset[15] == 5
    assert repeat_dataset[27] == 7
    assert len(repeat_dataset) == 10 * len(dataset_a)


def test_custom_dataset():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    crop_size = (512, 1024)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(128, 256), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(128, 256),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    # with img_dir and ann_dir
    train_dataset = CustomDataset(
        train_pipeline,
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        img_dir='imgs/',
        ann_dir='gts/',
        img_suffix='img.jpg',
        seg_map_suffix='gt.png')
    assert len(train_dataset) == 5

    # with img_dir, ann_dir, split
    train_dataset = CustomDataset(
        train_pipeline,
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        img_dir='imgs/',
        ann_dir='gts/',
        img_suffix='img.jpg',
        seg_map_suffix='gt.png',
        split='splits/train.txt')
    assert len(train_dataset) == 4

    # no data_root
    train_dataset = CustomDataset(
        train_pipeline,
        img_dir=osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs'),
        ann_dir=osp.join(osp.dirname(__file__), '../data/pseudo_dataset/gts'),
        img_suffix='img.jpg',
        seg_map_suffix='gt.png')
    assert len(train_dataset) == 5

    # with data_root but img_dir/ann_dir are abs path
    train_dataset = CustomDataset(
        train_pipeline,
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        img_dir=osp.abspath(
            osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs')),
        ann_dir=osp.abspath(
            osp.join(osp.dirname(__file__), '../data/pseudo_dataset/gts')),
        img_suffix='img.jpg',
        seg_map_suffix='gt.png')
    assert len(train_dataset) == 5

    # test_mode=True
    test_dataset = CustomDataset(
        test_pipeline,
        img_dir=osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs'),
        img_suffix='img.jpg',
        test_mode=True)
    assert len(test_dataset) == 5

    # training data get
    train_data = train_dataset[0]
    assert isinstance(train_data, dict)

    # test data get
    test_data = test_dataset[0]
    assert isinstance(test_data, dict)

    # get gt seg map
    gt_seg_maps = train_dataset.get_gt_seg_maps()
    assert len(gt_seg_maps) == 5

    # evaluation
    pseudo_results = []
    for gt_seg_map in gt_seg_maps:
        h, w = gt_seg_map.shape
        pseudo_results.append(np.random.randint(low=0, high=7, size=(h, w)))
    eval_results = train_dataset.evaluate(pseudo_results, metric=['mIoU'])
    assert isinstance(eval_results, dict)
    assert 'mIoU' in eval_results
    assert 'mAcc' in eval_results
    assert 'aAcc' in eval_results

    eval_results = train_dataset.evaluate(pseudo_results, metric='mDice')
    assert isinstance(eval_results, dict)
    assert 'mDice' in eval_results
    assert 'mAcc' in eval_results
    assert 'aAcc' in eval_results

    eval_results = train_dataset.evaluate(
        pseudo_results, metric=['mDice', 'mIoU'])
    assert isinstance(eval_results, dict)
    assert 'mIoU' in eval_results
    assert 'mDice' in eval_results
    assert 'mAcc' in eval_results
    assert 'aAcc' in eval_results

    # evaluation with CLASSES
    train_dataset.CLASSES = tuple(['a'] * 7)
    eval_results = train_dataset.evaluate(pseudo_results, metric='mIoU')
    assert isinstance(eval_results, dict)
    assert 'mIoU' in eval_results
    assert 'mAcc' in eval_results
    assert 'aAcc' in eval_results

    eval_results = train_dataset.evaluate(pseudo_results, metric='mDice')
    assert isinstance(eval_results, dict)
    assert 'mDice' in eval_results
    assert 'mAcc' in eval_results
    assert 'aAcc' in eval_results

    eval_results = train_dataset.evaluate(pseudo_results, metric='mFscore')
    assert isinstance(eval_results, dict)
    assert 'mRecall' in eval_results
    assert 'mPrecision' in eval_results
    assert 'mFscore' in eval_results
    assert 'aAcc' in eval_results

    eval_results = train_dataset.evaluate(
        pseudo_results, metric=['mIoU', 'mDice', 'mFscore'])
    assert isinstance(eval_results, dict)
    assert 'mIoU' in eval_results
    assert 'mDice' in eval_results
    assert 'mAcc' in eval_results
    assert 'aAcc' in eval_results
    assert 'mFscore' in eval_results
    assert 'mPrecision' in eval_results
    assert 'mRecall' in eval_results


@patch('mmseg.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmseg.datasets.CustomDataset.__getitem__',
       MagicMock(side_effect=lambda idx: idx))
@pytest.mark.parametrize('dataset, classes', [
    ('ADE20KDataset', ('wall', 'building')),
    ('CityscapesDataset', ('road', 'sidewalk')),
    ('CustomDataset', ('bus', 'car')),
    ('PascalVOCDataset', ('aeroplane', 'bicycle')),
])
def test_custom_classes_override_default(dataset, classes):

    dataset_class = DATASETS.get(dataset)

    original_classes = dataset_class.CLASSES

    # Test setting classes as a tuple
    custom_dataset = dataset_class(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        classes=classes,
        test_mode=True)

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == classes

    # Test setting classes as a list
    custom_dataset = dataset_class(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        classes=list(classes),
        test_mode=True)

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == list(classes)

    # Test overriding not a subset
    custom_dataset = dataset_class(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        classes=[classes[0]],
        test_mode=True)

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == [classes[0]]

    # Test default behavior
    custom_dataset = dataset_class(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        classes=None,
        test_mode=True)

    assert custom_dataset.CLASSES == original_classes


@patch('mmseg.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmseg.datasets.CustomDataset.__getitem__',
       MagicMock(side_effect=lambda idx: idx))
def test_custom_dataset_random_palette_is_generated():
    dataset = CustomDataset(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        classes=('bus', 'car'),
        test_mode=True)
    assert len(dataset.PALETTE) == 2
    for class_color in dataset.PALETTE:
        assert len(class_color) == 3
        assert all(x >= 0 and x <= 255 for x in class_color)


@patch('mmseg.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmseg.datasets.CustomDataset.__getitem__',
       MagicMock(side_effect=lambda idx: idx))
def test_custom_dataset_custom_palette():
    dataset = CustomDataset(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        classes=('bus', 'car'),
        palette=[[100, 100, 100], [200, 200, 200]],
        test_mode=True)
    assert tuple(dataset.PALETTE) == tuple([[100, 100, 100], [200, 200, 200]])
