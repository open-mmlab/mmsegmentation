# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch


def test_tversky_lose():
    from mmseg.models import build_loss

    # test alpha + beta != 1
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='TverskyLoss',
            class_weight=[1.0, 2.0, 3.0],
            loss_weight=1.0,
            alpha=0.4,
            beta=0.7,
            loss_name='loss_tversky')
        tversky_loss = build_loss(loss_cfg)
        logits = torch.rand(8, 3, 4, 4)
        labels = (torch.rand(8, 4, 4) * 3).long()
        tversky_loss(logits, labels, ignore_index=1)

    # test tversky loss
    loss_cfg = dict(
        type='TverskyLoss',
        class_weight=[1.0, 2.0, 3.0],
        loss_weight=1.0,
        ignore_index=1,
        loss_name='loss_tversky')
    tversky_loss = build_loss(loss_cfg)
    logits = torch.rand(8, 3, 4, 4)
    labels = (torch.rand(8, 4, 4) * 3).long()
    tversky_loss(logits, labels)

    # test loss with class weights from file
    import os
    import tempfile

    import mmengine
    import numpy as np
    tmp_file = tempfile.NamedTemporaryFile()

    mmengine.dump([1.0, 2.0, 3.0], f'{tmp_file.name}.pkl',
                  'pkl')  # from pkl file
    loss_cfg = dict(
        type='TverskyLoss',
        class_weight=f'{tmp_file.name}.pkl',
        loss_weight=1.0,
        ignore_index=1,
        loss_name='loss_tversky')
    tversky_loss = build_loss(loss_cfg)
    tversky_loss(logits, labels)

    np.save(f'{tmp_file.name}.npy', np.array([1.0, 2.0, 3.0]))  # from npy file
    loss_cfg = dict(
        type='TverskyLoss',
        class_weight=f'{tmp_file.name}.pkl',
        loss_weight=1.0,
        ignore_index=1,
        loss_name='loss_tversky')
    tversky_loss = build_loss(loss_cfg)
    tversky_loss(logits, labels)
    tmp_file.close()
    os.remove(f'{tmp_file.name}.pkl')
    os.remove(f'{tmp_file.name}.npy')

    # test tversky loss has name `loss_tversky`
    loss_cfg = dict(
        type='TverskyLoss',
        smooth=2,
        loss_weight=1.0,
        ignore_index=1,
        alpha=0.3,
        beta=0.7,
        loss_name='loss_tversky')
    tversky_loss = build_loss(loss_cfg)
    assert tversky_loss.loss_name == 'loss_tversky'
