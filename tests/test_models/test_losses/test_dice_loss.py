# Copyright (c) OpenMMLab. All rights reserved.
import torch


def test_dice_lose():
    from mmseg.models import build_loss

    # test dice loss with loss_type = 'multi_class'
    loss_cfg = dict(
        type='DiceLoss',
        reduction='none',
        class_weight=[1.0, 2.0, 3.0],
        loss_weight=1.0,
        ignore_index=1,
        loss_name='loss_dice')
    dice_loss = build_loss(loss_cfg)
    logits = torch.rand(8, 3, 4, 4)
    labels = (torch.rand(8, 4, 4) * 3).long()
    dice_loss(logits, labels)

    # test loss with class weights from file
    import os
    import tempfile
    import mmcv
    import numpy as np
    tmp_file = tempfile.NamedTemporaryFile()

    mmcv.dump([1.0, 2.0, 3.0], f'{tmp_file.name}.pkl', 'pkl')  # from pkl file
    loss_cfg = dict(
        type='DiceLoss',
        reduction='none',
        class_weight=f'{tmp_file.name}.pkl',
        loss_weight=1.0,
        ignore_index=1,
        loss_name='loss_dice')
    dice_loss = build_loss(loss_cfg)
    dice_loss(logits, labels, ignore_index=None)

    np.save(f'{tmp_file.name}.npy', np.array([1.0, 2.0, 3.0]))  # from npy file
    loss_cfg = dict(
        type='DiceLoss',
        reduction='none',
        class_weight=f'{tmp_file.name}.pkl',
        loss_weight=1.0,
        ignore_index=1,
        loss_name='loss_dice')
    dice_loss = build_loss(loss_cfg)
    dice_loss(logits, labels, ignore_index=None)
    tmp_file.close()
    os.remove(f'{tmp_file.name}.pkl')
    os.remove(f'{tmp_file.name}.npy')

    # test dice loss with loss_type = 'binary'
    loss_cfg = dict(
        type='DiceLoss',
        smooth=2,
        exponent=3,
        reduction='sum',
        loss_weight=1.0,
        ignore_index=0,
        loss_name='loss_dice')
    dice_loss = build_loss(loss_cfg)
    logits = torch.rand(8, 2, 4, 4)
    labels = (torch.rand(8, 4, 4) * 2).long()
    dice_loss(logits, labels)

    # test dice loss has name `loss_dice`
    loss_cfg = dict(
        type='DiceLoss',
        smooth=2,
        exponent=3,
        reduction='sum',
        loss_weight=1.0,
        ignore_index=0,
        loss_name='loss_dice')
    dice_loss = build_loss(loss_cfg)
    assert dice_loss.loss_name == 'loss_dice'
