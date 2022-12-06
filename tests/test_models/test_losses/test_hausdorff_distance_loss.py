# Copyright (c) OpenMMLab. All rights reserved.
import torch


def test_hausdorff_distance_loss():
    from mmseg.models import build_loss

    # test hausdorff distance loss
    loss_cfg = dict(
        type='HausdorffDistanceLoss',
        class_weight=[1.0, 2.0, 3.0],
        loss_weight=1.0,
        ignore_index=1,
        loss_name='loss_hausdorff_distance')
    hausdorff_distance_loss = build_loss(loss_cfg)
    logits = torch.rand(8, 3, 4, 4)
    labels = (torch.rand(8, 4, 4) * 3).long()
    hausdorff_distance_loss(logits, labels)

    # test hausdorff distance loss with class weights from file
    import os
    import tempfile

    import mmengine
    import numpy as np
    tmp_file = tempfile.NamedTemporaryFile()

    # from npy file
    mmengine.dump([1.0, 2.0, 3.0], f'{tmp_file.name}.pkl', 'pkl')
    loss_cfg = dict(
        type='HausdorffDistanceLoss',
        class_weight=f'{tmp_file.name}.pkl',
        loss_weight=1.0,
        ignore_index=1,
        loss_name='loss_hausdorff_distance')
    hausdorff_distance_loss = build_loss(loss_cfg)
    hausdorff_distance_loss(logits, labels)

    # from npy file
    np.save(f'{tmp_file.name}.npy', np.array([1.0, 2.0, 3.0]))
    loss_cfg = dict(
        type='HausdorffDistanceLoss',
        class_weight=f'{tmp_file.name}.npy',
        loss_weight=1.0,
        ignore_index=1,
        loss_name='loss_hausdorff_distance')
    hausdorff_distance_loss = build_loss(loss_cfg)
    hausdorff_distance_loss(logits, labels)
    tmp_file.close()
    os.remove(f'{tmp_file.name}.pkl')
    os.remove(f'{tmp_file.name}.npy')

    # test hausdorff distance loss has name `loss_hausdorff_distance`
    loss_cfg = dict(
        type='HausdorffDistanceLoss',
        loss_weight=1.0,
        ignore_index=1,
        loss_name='loss_hausdorff_distance')
    hausdorff_distance_loss = build_loss(loss_cfg)
    assert hausdorff_distance_loss.loss_name == 'loss_hausdorff_distance'
