from unittest.mock import MagicMock

from mmseg.core.evaluation import get_classes, get_palette
from mmseg.datasets import (ADEDataset, CityscapesDataset, ConcatDataset,
                            CustomDataset, PascalVOCDataset, RepeatDataset)


def test_classes():
    assert list(CityscapesDataset.CLASSES) == get_classes('cityscapes')
    assert list(PascalVOCDataset.CLASSES) == get_classes('voc') == get_classes(
        'pascal_voc')
    assert list(
        ADEDataset.CLASSES) == get_classes('ade') == get_classes('ade20k')


def test_palette():
    assert CityscapesDataset.PALETTE == get_palette('cityscapes')
    assert PascalVOCDataset.PALETTE == get_palette('voc') == get_palette(
        'pascal_voc')
    assert ADEDataset.PALETTE == get_palette('ade') == get_palette('ade20k')


def test_dataset_wrapper():
    CustomDataset.load_annotations = MagicMock()
    CustomDataset.__getitem__ = MagicMock(side_effect=lambda idx: idx)
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
