import torch


def test_dice_lose():
    from mmseg.models import build_loss

    # test dice loss with loss_type = 'multi_class'
    loss_cfg = dict(
        type='DiceLoss',
        reduction='none',
        class_weight=[1.0, 2.0, 3.0],
        loss_weight=1.0,
        ignore_index=1)
    dice_loss = build_loss(loss_cfg)
    logits = torch.rand(8, 3, 4, 4)
    labels = (torch.rand(8, 4, 4) * 3).long()
    loss = dice_loss(logits, labels).detach()

    # test loss with class weights from file
    import os
    import tempfile
    import mmcv
    import numpy as np
    tmp_file = tempfile.NamedTemporaryFile()

    mmcv.dump([1.0, 2.0, 3.0], f'{tmp_file}.pkl', 'pkl')  # from pkl file
    loss_cfg = dict(
        type='DiceLoss',
        reduction='none',
        class_weight=f'{tmp_file}.pkl',
        loss_weight=1.0,
        ignore_index=1)
    dice_loss = build_loss(loss_cfg)
    assert torch.allclose(dice_loss(logits, labels, ignore_index=None), loss)

    np.save(f'{tmp_file}.npy', np.array([1.0, 2.0, 3.0]))  # from npy file
    loss_cfg = dict(
        type='DiceLoss',
        reduction='none',
        class_weight=f'{tmp_file}.pkl',
        loss_weight=1.0,
        ignore_index=1)
    dice_loss = build_loss(loss_cfg)
    assert torch.allclose(dice_loss(logits, labels, ignore_index=None), loss)
    tmp_file.close()
    os.remove(f'{tmp_file}.pkl')
    os.remove(f'{tmp_file}.npy')

    # test dice loss with loss_type = 'binary'
    loss_cfg = dict(
        type='DiceLoss',
        smooth=2,
        exponent=3,
        reduction='sum',
        loss_weight=1.0,
        ignore_index=0)
    dice_loss = build_loss(loss_cfg)
    logits = torch.rand(8, 2, 4, 4)
    labels = (torch.rand(8, 4, 4) * 2).long()
    dice_loss(logits, labels)
