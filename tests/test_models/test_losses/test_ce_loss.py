# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.losses.cross_entropy_loss import _expand_onehot_labels


@pytest.mark.parametrize('use_sigmoid', [True, False])
@pytest.mark.parametrize('reduction', ('mean', 'sum', 'none'))
@pytest.mark.parametrize('avg_non_ignore', [True, False])
@pytest.mark.parametrize('bce_input_same_dim', [True, False])
def test_ce_loss(use_sigmoid, reduction, avg_non_ignore, bce_input_same_dim):
    from mmseg.models import build_loss

    # use_mask and use_sigmoid cannot be true at the same time
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='CrossEntropyLoss',
            use_mask=True,
            use_sigmoid=True,
            loss_weight=1.0)
        build_loss(loss_cfg)

    # test loss with simple case for ce/bce
    fake_pred = torch.Tensor([[100, -100]])
    fake_label = torch.Tensor([1]).long()
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=use_sigmoid,
        loss_weight=1.0,
        avg_non_ignore=avg_non_ignore,
        loss_name='loss_ce')
    loss_cls = build_loss(loss_cls_cfg)
    if use_sigmoid:
        assert torch.allclose(
            loss_cls(fake_pred, fake_label), torch.tensor(100.))
    else:
        assert torch.allclose(
            loss_cls(fake_pred, fake_label), torch.tensor(200.))

    # test loss with complicated case for ce/bce
    # when avg_non_ignore is False, `avg_factor` would not be calculated
    fake_pred = torch.full(size=(2, 21, 8, 8), fill_value=0.5)
    fake_label = torch.ones(2, 8, 8).long()
    fake_label[:, 0, 0] = 255
    fake_weight = None
    # extra test bce loss when pred.shape == label.shape
    if use_sigmoid and bce_input_same_dim:
        fake_pred = torch.randn(2, 10).float()
        fake_label = torch.rand(2, 10).float()
        fake_weight = torch.rand(2, 10)  # set weight in forward function
        fake_label[0, [1, 2, 5, 7]] = 255  # set ignore_index
        fake_label[1, [0, 5, 8, 9]] = 255
    loss_cls = build_loss(loss_cls_cfg)
    loss = loss_cls(
        fake_pred, fake_label, weight=fake_weight, ignore_index=255)
    if use_sigmoid:
        if fake_pred.dim() != fake_label.dim():
            fake_label, weight, valid_mask = _expand_onehot_labels(
                labels=fake_label,
                label_weights=None,
                target_shape=fake_pred.shape,
                ignore_index=255)
        else:
            # should mask out the ignored elements
            valid_mask = ((fake_label >= 0) & (fake_label != 255)).float()
            weight = valid_mask
        torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            fake_pred,
            fake_label.float(),
            reduction='none',
            weight=fake_weight)
        if avg_non_ignore:
            avg_factor = valid_mask.sum().item()
            torch_loss = (torch_loss * weight).sum() / avg_factor
        else:
            torch_loss = (torch_loss * weight).mean()
    else:
        if avg_non_ignore:
            torch_loss = torch.nn.functional.cross_entropy(
                fake_pred, fake_label, reduction='mean', ignore_index=255)
        else:
            torch_loss = torch.nn.functional.cross_entropy(
                fake_pred, fake_label, reduction='sum',
                ignore_index=255) / fake_label.numel()
    assert torch.allclose(loss, torch_loss)

    if use_sigmoid:
        # test loss with complicated case for ce/bce
        # when avg_non_ignore is False, `avg_factor` would not be calculated
        fake_pred = torch.full(size=(2, 21, 8, 8), fill_value=0.5)
        fake_label = torch.ones(2, 8, 8).long()
        fake_label[:, 0, 0] = 255
        fake_weight = torch.rand(2, 8, 8)

        loss_cls = build_loss(loss_cls_cfg)
        loss = loss_cls(
            fake_pred, fake_label, weight=fake_weight, ignore_index=255)
        if use_sigmoid:
            fake_label, weight, valid_mask = _expand_onehot_labels(
                labels=fake_label,
                label_weights=None,
                target_shape=fake_pred.shape,
                ignore_index=255)
            torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                fake_pred,
                fake_label.float(),
                reduction='none',
                weight=fake_weight.unsqueeze(1).expand(fake_pred.shape))
            if avg_non_ignore:
                avg_factor = valid_mask.sum().item()
                torch_loss = (torch_loss * weight).sum() / avg_factor
            else:
                torch_loss = (torch_loss * weight).mean()
        assert torch.allclose(loss, torch_loss)

    # test loss with class weights from file
    fake_pred = torch.Tensor([[100, -100]])
    fake_label = torch.Tensor([1]).long()
    import os
    import tempfile

    import mmcv
    import numpy as np
    tmp_file = tempfile.NamedTemporaryFile()

    mmcv.dump([0.8, 0.2], f'{tmp_file.name}.pkl', 'pkl')  # from pkl file
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=f'{tmp_file.name}.pkl',
        loss_weight=1.0,
        loss_name='loss_ce')
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(40.))

    np.save(f'{tmp_file.name}.npy', np.array([0.8, 0.2]))  # from npy file
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=f'{tmp_file.name}.npy',
        loss_weight=1.0,
        loss_name='loss_ce')
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(40.))
    tmp_file.close()
    os.remove(f'{tmp_file.name}.pkl')
    os.remove(f'{tmp_file.name}.npy')

    loss_cls_cfg = dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(200.))

    # test `avg_non_ignore`  without ignore index would not affect ce/bce loss
    # when reduction='sum'/'none'/'mean'
    loss_cls_cfg1 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=use_sigmoid,
        reduction=reduction,
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls1 = build_loss(loss_cls_cfg1)
    loss_cls_cfg2 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=use_sigmoid,
        reduction=reduction,
        loss_weight=1.0,
        avg_non_ignore=False)
    loss_cls2 = build_loss(loss_cls_cfg2)
    assert torch.allclose(
        loss_cls1(fake_pred, fake_label, ignore_index=255) / fake_pred.numel(),
        loss_cls2(fake_pred, fake_label, ignore_index=255) / fake_pred.numel(),
        atol=1e-4)

    # test ce/bce loss with ignore index and class weight
    # in 5-way classification
    if use_sigmoid:
        # test bce loss when pred.shape == or != label.shape
        if bce_input_same_dim:
            fake_pred = torch.randn(2, 10).float()
            fake_label = torch.rand(2, 10).float()
            class_weight = torch.rand(2, 10)
        else:
            fake_pred = torch.full(size=(2, 21, 8, 8), fill_value=0.5)
            fake_label = torch.ones(2, 8, 8).long()
            class_weight = torch.randn(2, 21, 8, 8)
            fake_label, weight, valid_mask = _expand_onehot_labels(
                labels=fake_label,
                label_weights=None,
                target_shape=fake_pred.shape,
                ignore_index=-100)
        torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            fake_pred,
            fake_label.float(),
            reduction='mean',
            pos_weight=class_weight)
    else:
        fake_pred = torch.randn(2, 5, 10).float()  # 5-way classification
        fake_label = torch.randint(0, 5, (2, 10)).long()
        class_weight = torch.rand(5)
        class_weight /= class_weight.sum()
        torch_loss = torch.nn.functional.cross_entropy(
            fake_pred, fake_label, reduction='sum',
            weight=class_weight) / fake_label.numel()
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=use_sigmoid,
        reduction='mean',
        class_weight=class_weight,
        loss_weight=1.0,
        avg_non_ignore=avg_non_ignore)
    loss_cls = build_loss(loss_cls_cfg)

    # test cross entropy loss has name `loss_ce`
    assert loss_cls.loss_name == 'loss_ce'
    # test avg_non_ignore is in extra_repr
    assert loss_cls.extra_repr() == f'avg_non_ignore={avg_non_ignore}'

    loss = loss_cls(fake_pred, fake_label)
    assert torch.allclose(loss, torch_loss)

    fake_label[0, [1, 2, 5, 7]] = 10  # set ignore_index
    fake_label[1, [0, 5, 8, 9]] = 10
    loss = loss_cls(fake_pred, fake_label, ignore_index=10)
    if use_sigmoid:
        if avg_non_ignore:
            torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                fake_pred[fake_label != 10],
                fake_label[fake_label != 10].float(),
                pos_weight=class_weight[fake_label != 10],
                reduction='mean')
        else:
            torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                fake_pred[fake_label != 10],
                fake_label[fake_label != 10].float(),
                pos_weight=class_weight[fake_label != 10],
                reduction='sum') / fake_label.numel()
    else:
        if avg_non_ignore:
            torch_loss = torch.nn.functional.cross_entropy(
                fake_pred,
                fake_label,
                ignore_index=10,
                reduction='sum',
                weight=class_weight) / fake_label[fake_label != 10].numel()
        else:
            torch_loss = torch.nn.functional.cross_entropy(
                fake_pred,
                fake_label,
                ignore_index=10,
                reduction='sum',
                weight=class_weight) / fake_label.numel()
    assert torch.allclose(loss, torch_loss)


@pytest.mark.parametrize('avg_non_ignore', [True, False])
@pytest.mark.parametrize('with_weight', [True, False])
def test_binary_class_ce_loss(avg_non_ignore, with_weight):
    from mmseg.models import build_loss

    fake_pred = torch.rand(3, 1, 10, 10)
    fake_label = torch.randint(0, 2, (3, 10, 10))
    fake_weight = torch.rand(3, 10, 10)
    valid_mask = ((fake_label >= 0) & (fake_label != 255)).float()
    weight = valid_mask

    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_pred,
        fake_label.unsqueeze(1).float(),
        reduction='none',
        weight=fake_weight.unsqueeze(1).float() if with_weight else None)
    if avg_non_ignore:
        eps = torch.finfo(torch.float32).eps
        avg_factor = valid_mask.sum().item()
        torch_loss = (torch_loss * weight.unsqueeze(1)).sum() / (
            avg_factor + eps)
    else:
        torch_loss = (torch_loss * weight.unsqueeze(1)).mean()

    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        loss_weight=1.0,
        avg_non_ignore=avg_non_ignore,
        reduction='mean',
        loss_name='loss_ce')
    loss_cls = build_loss(loss_cls_cfg)
    loss = loss_cls(
        fake_pred,
        fake_label,
        weight=fake_weight if with_weight else None,
        ignore_index=255)
    assert torch.allclose(loss, torch_loss)
