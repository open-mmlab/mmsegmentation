# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch


def test_lovasz_loss():
    from mmseg.models import build_loss

    # loss_type should be 'binary' or 'multi_class'
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='LovaszLoss',
            loss_type='Binary',
            reduction='none',
            loss_weight=1.0)
        build_loss(loss_cfg)

    # reduction should be 'none' when per_image is False.
    with pytest.raises(AssertionError):
        loss_cfg = dict(type='LovaszLoss', loss_type='multi_class')
        build_loss(loss_cfg)

    # test lovasz loss with loss_type = 'multi_class' and per_image = False
    loss_cfg = dict(type='LovaszLoss', reduction='none', loss_weight=1.0)
    lovasz_loss = build_loss(loss_cfg)
    logits = torch.rand(1, 3, 4, 4)
    labels = (torch.rand(1, 4, 4) * 2).long()
    lovasz_loss(logits, labels)

    # test lovasz loss with loss_type = 'multi_class' and per_image = True
    loss_cfg = dict(
        type='LovaszLoss',
        per_image=True,
        reduction='mean',
        class_weight=[1.0, 2.0, 3.0],
        loss_weight=1.0)
    lovasz_loss = build_loss(loss_cfg)
    logits = torch.rand(1, 3, 4, 4)
    labels = (torch.rand(1, 4, 4) * 2).long()
    lovasz_loss(logits, labels, ignore_index=None)

    # test loss with class weights from file
    import os
    import tempfile
    import mmcv
    import numpy as np
    tmp_file = tempfile.NamedTemporaryFile()

    mmcv.dump([1.0, 2.0, 3.0], f'{tmp_file.name}.pkl', 'pkl')  # from pkl file
    loss_cfg = dict(
        type='LovaszLoss',
        per_image=True,
        reduction='mean',
        class_weight=f'{tmp_file.name}.pkl',
        loss_weight=1.0)
    lovasz_loss = build_loss(loss_cfg)
    lovasz_loss(logits, labels, ignore_index=None)

    np.save(f'{tmp_file.name}.npy', np.array([1.0, 2.0, 3.0]))  # from npy file
    loss_cfg = dict(
        type='LovaszLoss',
        per_image=True,
        reduction='mean',
        class_weight=f'{tmp_file.name}.npy',
        loss_weight=1.0)
    lovasz_loss = build_loss(loss_cfg)
    lovasz_loss(logits, labels, ignore_index=None)
    tmp_file.close()
    os.remove(f'{tmp_file.name}.pkl')
    os.remove(f'{tmp_file.name}.npy')

    # test lovasz loss with loss_type = 'binary' and per_image = False
    loss_cfg = dict(
        type='LovaszLoss',
        loss_type='binary',
        reduction='none',
        loss_weight=1.0)
    lovasz_loss = build_loss(loss_cfg)
    logits = torch.rand(2, 4, 4)
    labels = (torch.rand(2, 4, 4)).long()
    lovasz_loss(logits, labels)

    # test lovasz loss with loss_type = 'binary' and per_image = True
    loss_cfg = dict(
        type='LovaszLoss',
        loss_type='binary',
        per_image=True,
        reduction='mean',
        loss_weight=1.0)
    lovasz_loss = build_loss(loss_cfg)
    logits = torch.rand(2, 4, 4)
    labels = (torch.rand(2, 4, 4)).long()
    lovasz_loss(logits, labels, ignore_index=None)
