import logging
import tempfile
from unittest.mock import MagicMock, patch

import mmcv.runner
import pytest
import torch
import torch.nn as nn
from mmcv.runner import obj_from_dict
from torch.utils.data import DataLoader, Dataset

from mmseg.apis import single_gpu_test
from mmseg.core import DistEvalHook, EvalHook


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

    def forward(self, img, img_metas, test_mode=False, **kwargs):
        return img

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)


def test_iter_eval_hook():
    with pytest.raises(TypeError):
        test_dataset = ExampleModel()
        data_loader = [
            DataLoader(
                test_dataset,
                batch_size=1,
                sampler=None,
                num_worker=0,
                shuffle=False)
        ]
        EvalHook(data_loader)

    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
    optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = obj_from_dict(optim_cfg, torch.optim,
                              dict(params=model.parameters()))

    # test EvalHook
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_hook = EvalHook(data_loader)
        runner = mmcv.runner.IterBasedRunner(
            model=model,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logging.getLogger())
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)
        test_dataset.evaluate.assert_called_with([torch.tensor([1])],
                                                 logger=runner.logger)


def test_epoch_eval_hook():
    with pytest.raises(TypeError):
        test_dataset = ExampleModel()
        data_loader = [
            DataLoader(
                test_dataset,
                batch_size=1,
                sampler=None,
                num_worker=0,
                shuffle=False)
        ]
        EvalHook(data_loader, by_epoch=True)

    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
    optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = obj_from_dict(optim_cfg, torch.optim,
                              dict(params=model.parameters()))

    # test EvalHook with interval
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_hook = EvalHook(data_loader, by_epoch=True, interval=2)
        runner = mmcv.runner.EpochBasedRunner(
            model=model,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logging.getLogger())
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 2)
        test_dataset.evaluate.assert_called_once_with([torch.tensor([1])],
                                                      logger=runner.logger)


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    results = single_gpu_test(model, data_loader)
    return results


@patch('mmseg.apis.multi_gpu_test', multi_gpu_test)
def test_dist_eval_hook():
    with pytest.raises(TypeError):
        test_dataset = ExampleModel()
        data_loader = [
            DataLoader(
                test_dataset,
                batch_size=1,
                sampler=None,
                num_worker=0,
                shuffle=False)
        ]
        DistEvalHook(data_loader)

    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
    optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = obj_from_dict(optim_cfg, torch.optim,
                              dict(params=model.parameters()))

    # test DistEvalHook
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_hook = DistEvalHook(data_loader)
        runner = mmcv.runner.IterBasedRunner(
            model=model,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logging.getLogger())
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)
        test_dataset.evaluate.assert_called_with([torch.tensor([1])],
                                                 logger=runner.logger)


@patch('mmseg.apis.multi_gpu_test', multi_gpu_test)
def test_dist_eval_hook_epoch():
    with pytest.raises(TypeError):
        test_dataset = ExampleModel()
        data_loader = [
            DataLoader(
                test_dataset,
                batch_size=1,
                sampler=None,
                num_worker=0,
                shuffle=False)
        ]
        DistEvalHook(data_loader)

    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
    optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = obj_from_dict(optim_cfg, torch.optim,
                              dict(params=model.parameters()))

    # test DistEvalHook
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_hook = DistEvalHook(data_loader, by_epoch=True, interval=2)
        runner = mmcv.runner.EpochBasedRunner(
            model=model,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logging.getLogger())
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 2)
        test_dataset.evaluate.assert_called_with([torch.tensor([1])],
                                                 logger=runner.logger)
