# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from mmseg.core.evaluation import get_classes, get_palette
from mmseg.datasets import (DATASETS, ADE20KDataset, CityscapesDataset,
                            COCOStuffDataset, ConcatDataset, CustomDataset,
                            ISPRSDataset, LoveDADataset, MultiImageMixDataset,
                            PascalVOCDataset, PotsdamDataset, RepeatDataset,
                            build_dataset, iSAIDDataset)


def test_classes():
    assert list(CityscapesDataset.CLASSES) == get_classes('cityscapes')
    assert list(PascalVOCDataset.CLASSES) == get_classes('voc') == get_classes(
        'pascal_voc')
    assert list(
        ADE20KDataset.CLASSES) == get_classes('ade') == get_classes('ade20k')
    assert list(COCOStuffDataset.CLASSES) == get_classes('cocostuff')
    assert list(LoveDADataset.CLASSES) == get_classes('loveda')
    assert list(PotsdamDataset.CLASSES) == get_classes('potsdam')
    assert list(ISPRSDataset.CLASSES) == get_classes('vaihingen')
    assert list(iSAIDDataset.CLASSES) == get_classes('isaid')

    with pytest.raises(ValueError):
        get_classes('unsupported')


def test_classes_file_path():
    tmp_file = tempfile.NamedTemporaryFile()
    classes_path = f'{tmp_file.name}.txt'
    train_pipeline = [dict(type='LoadImageFromFile')]
    kwargs = dict(pipeline=train_pipeline, img_dir='./', classes=classes_path)

    # classes.txt with full categories
    categories = get_classes('cityscapes')
    with open(classes_path, 'w') as f:
        f.write('\n'.join(categories))
    assert list(CityscapesDataset(**kwargs).CLASSES) == categories

    # classes.txt with sub categories
    categories = ['road', 'sidewalk', 'building']
    with open(classes_path, 'w') as f:
        f.write('\n'.join(categories))
    assert list(CityscapesDataset(**kwargs).CLASSES) == categories

    # classes.txt with unknown categories
    categories = ['road', 'sidewalk', 'unknown']
    with open(classes_path, 'w') as f:
        f.write('\n'.join(categories))

    with pytest.raises(ValueError):
        CityscapesDataset(**kwargs)

    tmp_file.close()
    os.remove(classes_path)
    assert not osp.exists(classes_path)


def test_palette():
    assert CityscapesDataset.PALETTE == get_palette('cityscapes')
    assert PascalVOCDataset.PALETTE == get_palette('voc') == get_palette(
        'pascal_voc')
    assert ADE20KDataset.PALETTE == get_palette('ade') == get_palette('ade20k')
    assert LoveDADataset.PALETTE == get_palette('loveda')
    assert PotsdamDataset.PALETTE == get_palette('potsdam')
    assert COCOStuffDataset.PALETTE == get_palette('cocostuff')
    assert iSAIDDataset.PALETTE == get_palette('isaid')

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

    img_scale = (60, 60)
    pipeline = [
        dict(type='RandomMosaic', prob=1, img_scale=img_scale),
        dict(type='RandomFlip', prob=0.5),
        dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    ]

    CustomDataset.load_annotations = MagicMock()
    results = []
    for _ in range(2):
        height = np.random.randint(10, 30)
        weight = np.random.randint(10, 30)
        img = np.ones((height, weight, 3))
        gt_semantic_seg = np.random.randint(5, size=(height, weight))
        results.append(dict(gt_semantic_seg=gt_semantic_seg, img=img))

    classes = ['0', '1', '2', '3', '4']
    palette = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]
    CustomDataset.__getitem__ = MagicMock(side_effect=lambda idx: results[idx])
    dataset_a = CustomDataset(
        img_dir=MagicMock(),
        pipeline=[],
        test_mode=True,
        classes=classes,
        palette=palette)
    len_a = 2
    dataset_a.img_infos = MagicMock()
    dataset_a.img_infos.__len__.return_value = len_a

    multi_image_mix_dataset = MultiImageMixDataset(dataset_a, pipeline)
    assert len(multi_image_mix_dataset) == len(dataset_a)

    for idx in range(len_a):
        results_ = multi_image_mix_dataset[idx]

    # test skip_type_keys
    multi_image_mix_dataset = MultiImageMixDataset(
        dataset_a, pipeline, skip_type_keys=('RandomFlip'))
    for idx in range(len_a):
        results_ = multi_image_mix_dataset[idx]
        assert results_['img'].shape == (img_scale[0], img_scale[1], 3)

    skip_type_keys = ('RandomFlip', 'Resize')
    multi_image_mix_dataset.update_skip_type_keys(skip_type_keys)
    for idx in range(len_a):
        results_ = multi_image_mix_dataset[idx]
        assert results_['img'].shape[:2] != img_scale

    # test pipeline
    with pytest.raises(TypeError):
        pipeline = [['Resize']]
        multi_image_mix_dataset = MultiImageMixDataset(dataset_a, pipeline)


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
        test_mode=True,
        classes=('pseudo_class', ))
    assert len(test_dataset) == 5

    # training data get
    train_data = train_dataset[0]
    assert isinstance(train_data, dict)

    # test data get
    test_data = test_dataset[0]
    assert isinstance(test_data, dict)

    # get gt seg map
    gt_seg_maps = train_dataset.get_gt_seg_maps(efficient_test=True)
    assert isinstance(gt_seg_maps, Generator)
    gt_seg_maps = list(gt_seg_maps)
    assert len(gt_seg_maps) == 5

    # format_results not implemented
    with pytest.raises(NotImplementedError):
        test_dataset.format_results([], '')

    pseudo_results = []
    for gt_seg_map in gt_seg_maps:
        h, w = gt_seg_map.shape
        pseudo_results.append(np.random.randint(low=0, high=7, size=(h, w)))

    # test past evaluation without CLASSES
    with pytest.raises(TypeError):
        eval_results = train_dataset.evaluate(pseudo_results, metric=['mIoU'])

    with pytest.raises(TypeError):
        eval_results = train_dataset.evaluate(pseudo_results, metric='mDice')

    with pytest.raises(TypeError):
        eval_results = train_dataset.evaluate(
            pseudo_results, metric=['mDice', 'mIoU'])

    # test past evaluation with CLASSES
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

    assert not np.isnan(eval_results['mIoU'])
    assert not np.isnan(eval_results['mDice'])
    assert not np.isnan(eval_results['mAcc'])
    assert not np.isnan(eval_results['aAcc'])
    assert not np.isnan(eval_results['mFscore'])
    assert not np.isnan(eval_results['mPrecision'])
    assert not np.isnan(eval_results['mRecall'])

    # test evaluation with pre-eval and the dataset.CLASSES is necessary
    train_dataset.CLASSES = tuple(['a'] * 7)
    pseudo_results = []
    for idx in range(len(train_dataset)):
        h, w = gt_seg_maps[idx].shape
        pseudo_result = np.random.randint(low=0, high=7, size=(h, w))
        pseudo_results.extend(train_dataset.pre_eval(pseudo_result, idx))
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

    assert not np.isnan(eval_results['mIoU'])
    assert not np.isnan(eval_results['mDice'])
    assert not np.isnan(eval_results['mAcc'])
    assert not np.isnan(eval_results['aAcc'])
    assert not np.isnan(eval_results['mFscore'])
    assert not np.isnan(eval_results['mPrecision'])
    assert not np.isnan(eval_results['mRecall'])


@pytest.mark.parametrize('separate_eval', [True, False])
def test_eval_concat_custom_dataset(separate_eval):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
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
    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_dataset')
    img_dir = 'imgs/'
    ann_dir = 'gts/'

    cfg1 = dict(
        type='CustomDataset',
        pipeline=test_pipeline,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        img_suffix='img.jpg',
        seg_map_suffix='gt.png',
        classes=tuple(['a'] * 7))
    dataset1 = build_dataset(cfg1)
    assert len(dataset1) == 5
    # get gt seg map
    gt_seg_maps = dataset1.get_gt_seg_maps(efficient_test=True)
    assert isinstance(gt_seg_maps, Generator)
    gt_seg_maps = list(gt_seg_maps)
    assert len(gt_seg_maps) == 5

    # test past evaluation
    pseudo_results = []
    for gt_seg_map in gt_seg_maps:
        h, w = gt_seg_map.shape
        pseudo_results.append(np.random.randint(low=0, high=7, size=(h, w)))
    eval_results1 = dataset1.evaluate(
        pseudo_results, metric=['mIoU', 'mDice', 'mFscore'])

    # We use same dir twice for simplicity
    # with ann_dir
    cfg2 = dict(
        type='CustomDataset',
        pipeline=test_pipeline,
        data_root=data_root,
        img_dir=[img_dir, img_dir],
        ann_dir=[ann_dir, ann_dir],
        img_suffix='img.jpg',
        seg_map_suffix='gt.png',
        classes=tuple(['a'] * 7),
        separate_eval=separate_eval)
    dataset2 = build_dataset(cfg2)
    assert isinstance(dataset2, ConcatDataset)
    assert len(dataset2) == 10

    eval_results2 = dataset2.evaluate(
        pseudo_results * 2, metric=['mIoU', 'mDice', 'mFscore'])

    if separate_eval:
        assert eval_results1['mIoU'] == eval_results2[
            '0_mIoU'] == eval_results2['1_mIoU']
        assert eval_results1['mDice'] == eval_results2[
            '0_mDice'] == eval_results2['1_mDice']
        assert eval_results1['mAcc'] == eval_results2[
            '0_mAcc'] == eval_results2['1_mAcc']
        assert eval_results1['aAcc'] == eval_results2[
            '0_aAcc'] == eval_results2['1_aAcc']
        assert eval_results1['mFscore'] == eval_results2[
            '0_mFscore'] == eval_results2['1_mFscore']
        assert eval_results1['mPrecision'] == eval_results2[
            '0_mPrecision'] == eval_results2['1_mPrecision']
        assert eval_results1['mRecall'] == eval_results2[
            '0_mRecall'] == eval_results2['1_mRecall']
    else:
        assert eval_results1['mIoU'] == eval_results2['mIoU']
        assert eval_results1['mDice'] == eval_results2['mDice']
        assert eval_results1['mAcc'] == eval_results2['mAcc']
        assert eval_results1['aAcc'] == eval_results2['aAcc']
        assert eval_results1['mFscore'] == eval_results2['mFscore']
        assert eval_results1['mPrecision'] == eval_results2['mPrecision']
        assert eval_results1['mRecall'] == eval_results2['mRecall']

    # test get dataset_idx and sample_idx from ConcateDataset
    dataset_idx, sample_idx = dataset2.get_dataset_idx_and_sample_idx(3)
    assert dataset_idx == 0
    assert sample_idx == 3

    dataset_idx, sample_idx = dataset2.get_dataset_idx_and_sample_idx(7)
    assert dataset_idx == 1
    assert sample_idx == 2

    dataset_idx, sample_idx = dataset2.get_dataset_idx_and_sample_idx(-7)
    assert dataset_idx == 0
    assert sample_idx == 3

    # test negative indice exceed length of dataset
    with pytest.raises(ValueError):
        dataset_idx, sample_idx = dataset2.get_dataset_idx_and_sample_idx(-11)

    # test negative indice value
    indice = -6
    dataset_idx1, sample_idx1 = dataset2.get_dataset_idx_and_sample_idx(indice)
    dataset_idx2, sample_idx2 = dataset2.get_dataset_idx_and_sample_idx(
        len(dataset2) + indice)
    assert dataset_idx1 == dataset_idx2
    assert sample_idx1 == sample_idx2

    # test evaluation with pre-eval and the dataset.CLASSES is necessary
    pseudo_results = []
    eval_results1 = []
    for idx in range(len(dataset1)):
        h, w = gt_seg_maps[idx].shape
        pseudo_result = np.random.randint(low=0, high=7, size=(h, w))
        pseudo_results.append(pseudo_result)
        eval_results1.extend(dataset1.pre_eval(pseudo_result, idx))

    assert len(eval_results1) == len(dataset1)
    assert isinstance(eval_results1[0], tuple)
    assert len(eval_results1[0]) == 4
    assert isinstance(eval_results1[0][0], torch.Tensor)

    eval_results1 = dataset1.evaluate(
        eval_results1, metric=['mIoU', 'mDice', 'mFscore'])

    pseudo_results = pseudo_results * 2
    eval_results2 = []
    for idx in range(len(dataset2)):
        eval_results2.extend(dataset2.pre_eval(pseudo_results[idx], idx))

    assert len(eval_results2) == len(dataset2)
    assert isinstance(eval_results2[0], tuple)
    assert len(eval_results2[0]) == 4
    assert isinstance(eval_results2[0][0], torch.Tensor)

    eval_results2 = dataset2.evaluate(
        eval_results2, metric=['mIoU', 'mDice', 'mFscore'])

    if separate_eval:
        assert eval_results1['mIoU'] == eval_results2[
            '0_mIoU'] == eval_results2['1_mIoU']
        assert eval_results1['mDice'] == eval_results2[
            '0_mDice'] == eval_results2['1_mDice']
        assert eval_results1['mAcc'] == eval_results2[
            '0_mAcc'] == eval_results2['1_mAcc']
        assert eval_results1['aAcc'] == eval_results2[
            '0_aAcc'] == eval_results2['1_aAcc']
        assert eval_results1['mFscore'] == eval_results2[
            '0_mFscore'] == eval_results2['1_mFscore']
        assert eval_results1['mPrecision'] == eval_results2[
            '0_mPrecision'] == eval_results2['1_mPrecision']
        assert eval_results1['mRecall'] == eval_results2[
            '0_mRecall'] == eval_results2['1_mRecall']
    else:
        assert eval_results1['mIoU'] == eval_results2['mIoU']
        assert eval_results1['mDice'] == eval_results2['mDice']
        assert eval_results1['mAcc'] == eval_results2['mAcc']
        assert eval_results1['aAcc'] == eval_results2['aAcc']
        assert eval_results1['mFscore'] == eval_results2['mFscore']
        assert eval_results1['mPrecision'] == eval_results2['mPrecision']
        assert eval_results1['mRecall'] == eval_results2['mRecall']

    # test batch_indices for pre eval
    eval_results2 = dataset2.pre_eval(pseudo_results,
                                      list(range(len(pseudo_results))))

    assert len(eval_results2) == len(dataset2)
    assert isinstance(eval_results2[0], tuple)
    assert len(eval_results2[0]) == 4
    assert isinstance(eval_results2[0][0], torch.Tensor)

    eval_results2 = dataset2.evaluate(
        eval_results2, metric=['mIoU', 'mDice', 'mFscore'])

    if separate_eval:
        assert eval_results1['mIoU'] == eval_results2[
            '0_mIoU'] == eval_results2['1_mIoU']
        assert eval_results1['mDice'] == eval_results2[
            '0_mDice'] == eval_results2['1_mDice']
        assert eval_results1['mAcc'] == eval_results2[
            '0_mAcc'] == eval_results2['1_mAcc']
        assert eval_results1['aAcc'] == eval_results2[
            '0_aAcc'] == eval_results2['1_aAcc']
        assert eval_results1['mFscore'] == eval_results2[
            '0_mFscore'] == eval_results2['1_mFscore']
        assert eval_results1['mPrecision'] == eval_results2[
            '0_mPrecision'] == eval_results2['1_mPrecision']
        assert eval_results1['mRecall'] == eval_results2[
            '0_mRecall'] == eval_results2['1_mRecall']
    else:
        assert eval_results1['mIoU'] == eval_results2['mIoU']
        assert eval_results1['mDice'] == eval_results2['mDice']
        assert eval_results1['mAcc'] == eval_results2['mAcc']
        assert eval_results1['aAcc'] == eval_results2['aAcc']
        assert eval_results1['mFscore'] == eval_results2['mFscore']
        assert eval_results1['mPrecision'] == eval_results2['mPrecision']
        assert eval_results1['mRecall'] == eval_results2['mRecall']


def test_ade():
    test_dataset = ADE20KDataset(
        pipeline=[],
        img_dir=osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs'))
    assert len(test_dataset) == 5

    # Test format_results
    pseudo_results = []
    for _ in range(len(test_dataset)):
        h, w = (2, 2)
        pseudo_results.append(np.random.randint(low=0, high=7, size=(h, w)))

    file_paths = test_dataset.format_results(pseudo_results, '.format_ade')
    assert len(file_paths) == len(test_dataset)
    temp = np.array(Image.open(file_paths[0]))
    assert np.allclose(temp, pseudo_results[0] + 1)

    shutil.rmtree('.format_ade')


@pytest.mark.parametrize('separate_eval', [True, False])
def test_concat_ade(separate_eval):
    test_dataset = ADE20KDataset(
        pipeline=[],
        img_dir=osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs'))
    assert len(test_dataset) == 5

    concat_dataset = ConcatDataset([test_dataset, test_dataset],
                                   separate_eval=separate_eval)
    assert len(concat_dataset) == 10
    # Test format_results
    pseudo_results = []
    for _ in range(len(concat_dataset)):
        h, w = (2, 2)
        pseudo_results.append(np.random.randint(low=0, high=7, size=(h, w)))

    # test format per image
    file_paths = []
    for i in range(len(pseudo_results)):
        file_paths.extend(
            concat_dataset.format_results([pseudo_results[i]],
                                          '.format_ade',
                                          indices=[i]))
    assert len(file_paths) == len(concat_dataset)
    temp = np.array(Image.open(file_paths[0]))
    assert np.allclose(temp, pseudo_results[0] + 1)

    shutil.rmtree('.format_ade')

    # test default argument
    file_paths = concat_dataset.format_results(pseudo_results, '.format_ade')
    assert len(file_paths) == len(concat_dataset)
    temp = np.array(Image.open(file_paths[0]))
    assert np.allclose(temp, pseudo_results[0] + 1)

    shutil.rmtree('.format_ade')


def test_cityscapes():
    test_dataset = CityscapesDataset(
        pipeline=[],
        img_dir=osp.join(
            osp.dirname(__file__),
            '../data/pseudo_cityscapes_dataset/leftImg8bit'),
        ann_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_cityscapes_dataset/gtFine'))
    assert len(test_dataset) == 1

    gt_seg_maps = list(test_dataset.get_gt_seg_maps())

    # Test format_results
    pseudo_results = []
    for idx in range(len(test_dataset)):
        h, w = gt_seg_maps[idx].shape
        pseudo_results.append(np.random.randint(low=0, high=19, size=(h, w)))

    file_paths = test_dataset.format_results(pseudo_results, '.format_city')
    assert len(file_paths) == len(test_dataset)
    temp = np.array(Image.open(file_paths[0]))
    assert np.allclose(temp,
                       test_dataset._convert_to_label_id(pseudo_results[0]))

    # Test cityscapes evaluate

    test_dataset.evaluate(
        pseudo_results, metric='cityscapes', imgfile_prefix='.format_city')

    shutil.rmtree('.format_city')


@pytest.mark.parametrize('separate_eval', [True, False])
def test_concat_cityscapes(separate_eval):
    cityscape_dataset = CityscapesDataset(
        pipeline=[],
        img_dir=osp.join(
            osp.dirname(__file__),
            '../data/pseudo_cityscapes_dataset/leftImg8bit'),
        ann_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_cityscapes_dataset/gtFine'))
    assert len(cityscape_dataset) == 1
    with pytest.raises(NotImplementedError):
        _ = ConcatDataset([cityscape_dataset, cityscape_dataset],
                          separate_eval=separate_eval)
    ade_dataset = ADE20KDataset(
        pipeline=[],
        img_dir=osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs'))
    assert len(ade_dataset) == 5
    with pytest.raises(NotImplementedError):
        _ = ConcatDataset([cityscape_dataset, ade_dataset],
                          separate_eval=separate_eval)


def test_loveda():
    test_dataset = LoveDADataset(
        pipeline=[],
        img_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_loveda_dataset/img_dir'),
        ann_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_loveda_dataset/ann_dir'))
    assert len(test_dataset) == 3

    gt_seg_maps = list(test_dataset.get_gt_seg_maps())

    # Test format_results
    pseudo_results = []
    for idx in range(len(test_dataset)):
        h, w = gt_seg_maps[idx].shape
        pseudo_results.append(np.random.randint(low=0, high=7, size=(h, w)))
    file_paths = test_dataset.format_results(pseudo_results, '.format_loveda')
    assert len(file_paths) == len(test_dataset)
    # Test loveda evaluate

    test_dataset.evaluate(
        pseudo_results, metric='mIoU', imgfile_prefix='.format_loveda')

    shutil.rmtree('.format_loveda')


def test_potsdam():
    test_dataset = PotsdamDataset(
        pipeline=[],
        img_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_potsdam_dataset/img_dir'),
        ann_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_potsdam_dataset/ann_dir'))
    assert len(test_dataset) == 1


def test_vaihingen():
    test_dataset = ISPRSDataset(
        pipeline=[],
        img_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_vaihingen_dataset/img_dir'),
        ann_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_vaihingen_dataset/ann_dir'))
    assert len(test_dataset) == 1


def test_isaid():
    test_dataset = iSAIDDataset(
        pipeline=[],
        img_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_isaid_dataset/img_dir'),
        ann_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_isaid_dataset/ann_dir'))
    assert len(test_dataset) == 2
    isaid_info = test_dataset.load_annotations(
        img_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_isaid_dataset/img_dir'),
        img_suffix='.png',
        ann_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_isaid_dataset/ann_dir'),
        seg_map_suffix='.png',
        split=osp.join(
            osp.dirname(__file__),
            '../data/pseudo_isaid_dataset/splits/train.txt'))
    assert len(isaid_info) == 1


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
    if dataset_class is CustomDataset:
        with pytest.raises(AssertionError):
            custom_dataset = dataset_class(
                pipeline=[],
                img_dir=MagicMock(),
                split=MagicMock(),
                classes=None,
                test_mode=True)
    else:
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
