import torch


def test_tversky_lose():
    from mmseg.models import build_loss

    # test tversky loss with loss_type = 'multi_class'
    loss_cfg = dict(
        type='TverskyLoss',
        reduction='none',
        class_weight=[1.0, 2.0, 3.0],
        loss_weight=1.0,
        ignore_index=1)
    tversky_loss = build_loss(loss_cfg)
    logits = torch.rand(8, 3, 4, 4)
    labels = (torch.rand(8, 4, 4) * 3).long()
    tversky_loss(logits, labels)

    # test loss with class weights from file
    import os
    import tempfile
    import mmcv
    import numpy as np
    tmp_file = tempfile.NamedTemporaryFile()

    mmcv.dump([1.0, 2.0, 3.0], f'{tmp_file.name}.pkl', 'pkl')  # from pkl file
    loss_cfg = dict(
        type='TverskyLoss',
        reduction='none',
        class_weight=f'{tmp_file.name}.pkl',
        loss_weight=1.0,
        ignore_index=1)
    tversky_loss = build_loss(loss_cfg)
    tversky_loss(logits, labels, ignore_index=None)

    np.save(f'{tmp_file.name}.npy', np.array([1.0, 2.0, 3.0]))  # from npy file
    loss_cfg = dict(
        type='TverskyLoss',
        reduction='none',
        class_weight=f'{tmp_file.name}.pkl',
        loss_weight=1.0,
        ignore_index=1)
    tversky_loss = build_loss(loss_cfg)
    tversky_loss(logits, labels, ignore_index=None)
    tmp_file.close()
    os.remove(f'{tmp_file.name}.pkl')
    os.remove(f'{tmp_file.name}.npy')

    # test tversky loss with loss_type = 'binary'
    loss_cfg = dict(
        type='TverskyLoss',
        smooth=2,
        exponent=3,
        reduction='sum',
        loss_weight=1.0,
        ignore_index=0,
        alpha=0.3,
        beta=0.7)
    tversky_loss = build_loss(loss_cfg)
    logits = torch.rand(8, 2, 4, 4)
    labels = (torch.rand(8, 4, 4) * 2).long()
    tversky_loss(logits, labels)
