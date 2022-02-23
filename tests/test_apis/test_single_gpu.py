# Copyright (c) OpenMMLab. All rights reserved.
import shutil
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, dataloader

from mmseg.apis import single_gpu_test


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([1]), img_metas=dict())
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, img, img_metas, return_loss=False, **kwargs):
        return img


def test_single_gpu():
    test_dataset = ExampleDataset()
    data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False,
    )
    model = ExampleModel()

    # Test efficient test compatibility (will be deprecated)
    results = single_gpu_test(model, data_loader, efficient_test=True)
    assert len(results) == 1
    pred = np.load(results[0])
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (1, )
    assert pred[0] == 1

    shutil.rmtree('.efficient_test')

    # Test pre_eval
    test_dataset.pre_eval = MagicMock(return_value=['success'])
    results = single_gpu_test(model, data_loader, pre_eval=True)
    assert results == ['success']

    # Test format_only
    test_dataset.format_results = MagicMock(return_value=['success'])
    results = single_gpu_test(model, data_loader, format_only=True)
    assert results == ['success']

    # efficient_test, pre_eval and format_only are mutually exclusive
    with pytest.raises(AssertionError):
        single_gpu_test(
            model,
            dataloader,
            efficient_test=True,
            format_only=True,
            pre_eval=True)
