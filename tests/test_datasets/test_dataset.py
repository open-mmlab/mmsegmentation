# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile

import pytest

from mmseg.datasets import (ADE20KDataset, BaseSegDataset, CityscapesDataset,
                            COCOStuffDataset, DecathlonDataset, DSDLSegDataset,
                            ISPRSDataset, LIPDataset, LoveDADataset,
                            MapillaryDataset_v1, MapillaryDataset_v2,
                            PascalVOCDataset, PotsdamDataset, REFUGEDataset,
                            SynapseDataset, iSAIDDataset)
from mmseg.registry import DATASETS
from mmseg.utils import get_classes, get_palette

try:
    from dsdl.dataset import DSDLDataset
except ImportError:
    DSDLDataset = None


def test_classes():
    assert list(
        CityscapesDataset.METAINFO['classes']) == get_classes('cityscapes')
    assert list(PascalVOCDataset.METAINFO['classes']) == get_classes(
        'voc') == get_classes('pascal_voc')
    assert list(ADE20KDataset.METAINFO['classes']) == get_classes(
        'ade') == get_classes('ade20k')
    assert list(
        COCOStuffDataset.METAINFO['classes']) == get_classes('cocostuff')
    assert list(LoveDADataset.METAINFO['classes']) == get_classes('loveda')
    assert list(PotsdamDataset.METAINFO['classes']) == get_classes('potsdam')
    assert list(ISPRSDataset.METAINFO['classes']) == get_classes('vaihingen')
    assert list(iSAIDDataset.METAINFO['classes']) == get_classes('isaid')
    assert list(
        MapillaryDataset_v1.METAINFO['classes']) == get_classes('mapillary_v1')
    assert list(
        MapillaryDataset_v2.METAINFO['classes']) == get_classes('mapillary_v2')

    with pytest.raises(ValueError):
        get_classes('unsupported')


def test_classes_file_path():
    tmp_file = tempfile.NamedTemporaryFile()
    classes_path = f'{tmp_file.name}.txt'
    train_pipeline = []
    kwargs = dict(
        pipeline=train_pipeline,
        data_prefix=dict(img_path='./', seg_map_path='./'),
        metainfo=dict(classes=classes_path))

    # classes.txt with full categories
    categories = get_classes('cityscapes')
    with open(classes_path, 'w') as f:
        f.write('\n'.join(categories))
    dataset = CityscapesDataset(**kwargs)
    assert list(dataset.metainfo['classes']) == categories
    assert dataset.label_map is None

    # classes.txt with sub categories
    categories = ['road', 'sidewalk', 'building']
    with open(classes_path, 'w') as f:
        f.write('\n'.join(categories))
    dataset = CityscapesDataset(**kwargs)
    assert list(dataset.metainfo['classes']) == categories
    assert dataset.label_map is not None

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
    assert CityscapesDataset.METAINFO['palette'] == get_palette('cityscapes')
    assert PascalVOCDataset.METAINFO['palette'] == get_palette(
        'voc') == get_palette('pascal_voc')
    assert ADE20KDataset.METAINFO['palette'] == get_palette(
        'ade') == get_palette('ade20k')
    assert LoveDADataset.METAINFO['palette'] == get_palette('loveda')
    assert PotsdamDataset.METAINFO['palette'] == get_palette('potsdam')
    assert COCOStuffDataset.METAINFO['palette'] == get_palette('cocostuff')
    assert iSAIDDataset.METAINFO['palette'] == get_palette('isaid')
    assert list(
        MapillaryDataset_v1.METAINFO['palette']) == get_palette('mapillary_v1')
    assert list(
        MapillaryDataset_v2.METAINFO['palette']) == get_palette('mapillary_v2')

    with pytest.raises(ValueError):
        get_palette('unsupported')


def test_custom_dataset():

    # with 'img_path' and 'seg_map_path' in data_prefix
    train_dataset = BaseSegDataset(
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        data_prefix=dict(
            img_path='imgs/',
            seg_map_path='gts/',
        ),
        img_suffix='img.jpg',
        seg_map_suffix='gt.png')
    assert len(train_dataset) == 5

    # with 'img_path' and 'seg_map_path' in data_prefix and ann_file
    train_dataset = BaseSegDataset(
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        data_prefix=dict(
            img_path='imgs/',
            seg_map_path='gts/',
        ),
        img_suffix='img.jpg',
        seg_map_suffix='gt.png',
        ann_file='splits/train.txt')
    assert len(train_dataset) == 4

    # no data_root
    train_dataset = BaseSegDataset(
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__), '../data/pseudo_dataset/imgs'),
            seg_map_path=osp.join(
                osp.dirname(__file__), '../data/pseudo_dataset/gts')),
        img_suffix='img.jpg',
        seg_map_suffix='gt.png')
    assert len(train_dataset) == 5

    # with data_root but 'img_path' and 'seg_map_path' in data_prefix are
    # abs path
    train_dataset = BaseSegDataset(
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__), '../data/pseudo_dataset/imgs'),
            seg_map_path=osp.join(
                osp.dirname(__file__), '../data/pseudo_dataset/gts')),
        img_suffix='img.jpg',
        seg_map_suffix='gt.png')
    assert len(train_dataset) == 5

    # test_mode=True
    test_dataset = BaseSegDataset(
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__), '../data/pseudo_dataset/imgs')),
        img_suffix='img.jpg',
        test_mode=True,
        metainfo=dict(classes=('pseudo_class', )))
    assert len(test_dataset) == 5

    # training data get
    train_data = train_dataset[0]
    assert isinstance(train_data, dict)
    assert 'img_path' in train_data and osp.isfile(train_data['img_path'])
    assert 'seg_map_path' in train_data and osp.isfile(
        train_data['seg_map_path'])

    # test data get
    test_data = test_dataset[0]
    assert isinstance(test_data, dict)
    assert 'img_path' in train_data and osp.isfile(train_data['img_path'])
    assert 'seg_map_path' in train_data and osp.isfile(
        train_data['seg_map_path'])


def test_ade():
    test_dataset = ADE20KDataset(
        pipeline=[],
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__), '../data/pseudo_dataset/imgs')))
    assert len(test_dataset) == 5


def test_cityscapes():
    test_dataset = CityscapesDataset(
        pipeline=[],
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_cityscapes_dataset/leftImg8bit/val'),
            seg_map_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_cityscapes_dataset/gtFine/val')))
    assert len(test_dataset) == 1


def test_loveda():
    test_dataset = LoveDADataset(
        pipeline=[],
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_loveda_dataset/img_dir'),
            seg_map_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_loveda_dataset/ann_dir')))
    assert len(test_dataset) == 3


def test_potsdam():
    test_dataset = PotsdamDataset(
        pipeline=[],
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_potsdam_dataset/img_dir'),
            seg_map_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_potsdam_dataset/ann_dir')))
    assert len(test_dataset) == 1


def test_vaihingen():
    test_dataset = ISPRSDataset(
        pipeline=[],
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_vaihingen_dataset/img_dir'),
            seg_map_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_vaihingen_dataset/ann_dir')))
    assert len(test_dataset) == 1


def test_synapse():
    test_dataset = SynapseDataset(
        pipeline=[],
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_synapse_dataset/img_dir'),
            seg_map_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_synapse_dataset/ann_dir')))
    assert len(test_dataset) == 2


def test_refuge():
    test_dataset = REFUGEDataset(
        pipeline=[],
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_refuge_dataset/img_dir'),
            seg_map_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_refuge_dataset/ann_dir')))
    assert len(test_dataset) == 1


def test_isaid():
    test_dataset = iSAIDDataset(
        pipeline=[],
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__), '../data/pseudo_isaid_dataset/img_dir'),
            seg_map_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_isaid_dataset/ann_dir')))
    assert len(test_dataset) == 2
    test_dataset = iSAIDDataset(
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__), '../data/pseudo_isaid_dataset/img_dir'),
            seg_map_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_isaid_dataset/ann_dir')),
        ann_file=osp.join(
            osp.dirname(__file__),
            '../data/pseudo_isaid_dataset/splits/train.txt'))
    assert len(test_dataset) == 1


def test_decathlon():
    data_root = osp.join(osp.dirname(__file__), '../data')
    # test load training dataset
    test_dataset = DecathlonDataset(
        pipeline=[], data_root=data_root, ann_file='dataset.json')
    assert len(test_dataset) == 1

    # test load test dataset
    test_dataset = DecathlonDataset(
        pipeline=[],
        data_root=data_root,
        ann_file='dataset.json',
        test_mode=True)
    assert len(test_dataset) == 3


def test_lip():
    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_lip_dataset')
    # train load training dataset
    train_dataset = LIPDataset(
        pipeline=[],
        data_root=data_root,
        data_prefix=dict(
            img_path='train_images', seg_map_path='train_segmentations'))
    assert len(train_dataset) == 1

    # test load training dataset
    test_dataset = LIPDataset(
        pipeline=[],
        data_root=data_root,
        data_prefix=dict(
            img_path='val_images', seg_map_path='val_segmentations'))
    assert len(test_dataset) == 1


def test_mapillary():
    test_dataset = MapillaryDataset_v1(
        pipeline=[],
        data_prefix=dict(
            img_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_mapillary_dataset/images'),
            seg_map_path=osp.join(
                osp.dirname(__file__),
                '../data/pseudo_mapillary_dataset/v1.2')))
    assert len(test_dataset) == 1


@pytest.mark.parametrize('dataset, classes', [
    ('ADE20KDataset', ('wall', 'building')),
    ('CityscapesDataset', ('road', 'sidewalk')),
    ('BaseSegDataset', ('bus', 'car')),
    ('PascalVOCDataset', ('aeroplane', 'bicycle')),
])
def test_custom_classes_override_default(dataset, classes):

    dataset_class = DATASETS.get(dataset)
    original_classes = dataset_class.METAINFO.get('classes', None)

    tmp_file = tempfile.NamedTemporaryFile()
    ann_file = tmp_file.name
    img_path = tempfile.mkdtemp()

    # Test setting classes as a tuple
    custom_dataset = dataset_class(
        data_prefix=dict(img_path=img_path),
        ann_file=ann_file,
        metainfo=dict(classes=classes),
        test_mode=True,
        lazy_init=True)

    assert custom_dataset.metainfo['classes'] != original_classes
    assert custom_dataset.metainfo['classes'] == classes
    if not isinstance(custom_dataset, BaseSegDataset):
        assert isinstance(custom_dataset.label_map, dict)

    # Test setting classes as a list
    custom_dataset = dataset_class(
        data_prefix=dict(img_path=img_path),
        ann_file=ann_file,
        metainfo=dict(classes=list(classes)),
        test_mode=True,
        lazy_init=True)

    assert custom_dataset.metainfo['classes'] != original_classes
    assert custom_dataset.metainfo['classes'] == list(classes)
    if not isinstance(custom_dataset, BaseSegDataset):
        assert isinstance(custom_dataset.label_map, dict)

    # Test overriding not a subset
    custom_dataset = dataset_class(
        ann_file=ann_file,
        data_prefix=dict(img_path=img_path),
        metainfo=dict(classes=[classes[0]]),
        test_mode=True,
        lazy_init=True)

    assert custom_dataset.metainfo['classes'] != original_classes
    assert custom_dataset.metainfo['classes'] == [classes[0]]
    if not isinstance(custom_dataset, BaseSegDataset):
        assert isinstance(custom_dataset.label_map, dict)

    # Test default behavior
    if dataset_class is BaseSegDataset:
        with pytest.raises(AssertionError):
            custom_dataset = dataset_class(
                ann_file=ann_file,
                data_prefix=dict(img_path=img_path),
                metainfo=None,
                test_mode=True,
                lazy_init=True)
    else:
        custom_dataset = dataset_class(
            data_prefix=dict(img_path=img_path),
            ann_file=ann_file,
            metainfo=None,
            test_mode=True,
            lazy_init=True)

        assert custom_dataset.METAINFO['classes'] == original_classes
        assert custom_dataset.label_map is None


def test_custom_dataset_random_palette_is_generated():
    dataset = BaseSegDataset(
        pipeline=[],
        data_prefix=dict(img_path=tempfile.mkdtemp()),
        ann_file=tempfile.mkdtemp(),
        metainfo=dict(classes=('bus', 'car')),
        lazy_init=True,
        test_mode=True)
    assert len(dataset.metainfo['palette']) == 2
    for class_color in dataset.metainfo['palette']:
        assert len(class_color) == 3
        assert all(x >= 0 and x <= 255 for x in class_color)


def test_custom_dataset_custom_palette():
    dataset = BaseSegDataset(
        data_prefix=dict(img_path=tempfile.mkdtemp()),
        ann_file=tempfile.mkdtemp(),
        metainfo=dict(
            classes=('bus', 'car'), palette=[[100, 100, 100], [200, 200,
                                                               200]]),
        lazy_init=True,
        test_mode=True)
    assert tuple(dataset.metainfo['palette']) == tuple([[100, 100, 100],
                                                        [200, 200, 200]])
    # test custom class and palette don't match
    with pytest.raises(ValueError):
        dataset = BaseSegDataset(
            data_prefix=dict(img_path=tempfile.mkdtemp()),
            ann_file=tempfile.mkdtemp(),
            metainfo=dict(classes=('bus', 'car'), palette=[[200, 200, 200]]),
            lazy_init=True)


def test_dsdlseg_dataset():
    if DSDLDataset is not None:
        dataset = DSDLSegDataset(
            data_root='tests/data/dsdl_seg', ann_file='set-train/train.yaml')
        assert len(dataset) == 3
        assert len(dataset.metainfo['classes']) == 21
    else:
        ImportWarning('Package `dsdl` is not installed.')
