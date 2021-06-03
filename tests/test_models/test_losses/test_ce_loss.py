import pytest
import torch


def test_ce_loss():
    from mmseg.models import build_loss

    # use_mask and use_sigmoid cannot be true at the same time
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='CrossEntropyLoss',
            use_mask=True,
            use_sigmoid=True,
            loss_weight=1.0)
        build_loss(loss_cfg)

    # test loss with class weights
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=[0.8, 0.2],
        loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    fake_pred = torch.Tensor([[100, -100]])
    fake_label = torch.Tensor([1]).long()
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(40.))

    # test loss with class weights from file
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
        loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(40.))

    np.save(f'{tmp_file.name}.npy', np.array([0.8, 0.2]))  # from npy file
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=f'{tmp_file.name}.npy',
        loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(40.))
    tmp_file.close()
    os.remove(f'{tmp_file.name}.pkl')
    os.remove(f'{tmp_file.name}.npy')

    loss_cls_cfg = dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(200.))

    loss_cls_cfg = dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(100.))

    fake_pred = torch.full(size=(2, 21, 8, 8), fill_value=0.5)
    fake_label = torch.ones(2, 8, 8).long()
    assert torch.allclose(
        loss_cls(fake_pred, fake_label), torch.tensor(0.9503), atol=1e-4)
    fake_label[:, 0, 0] = 255
    assert torch.allclose(
        loss_cls(fake_pred, fake_label, ignore_index=255),
        torch.tensor(0.9354),
        atol=1e-4)

    # test ce loss with ignore index
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='mean',
        class_weight=None,
        loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)

    fake_pred = torch.randn(2, 5, 10).float()  # 5-way classification
    fake_label = torch.randint(0, 5, (2, 10)).long()
    loss = loss_cls(fake_pred, fake_label)
    torch_loss = torch.nn.functional.cross_entropy(
        fake_pred, fake_label, reduction='mean')
    assert torch.allclose(loss, torch_loss)

    fake_label[0, [1, 2, 5, 7]] = 10  # set ignore_index
    fake_label[1, [0, 5, 8, 9]] = 10
    loss = loss_cls(fake_pred, fake_label, ignore_index=10)
    torch_loss = torch.nn.functional.cross_entropy(
        fake_pred, fake_label, ignore_index=10, reduction='mean')
    assert torch.allclose(loss, torch_loss)

    # test ignore index and class_weight
    class_weight = torch.rand(5)
    class_weight /= class_weight.sum()
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='mean',
        class_weight=class_weight,
        loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    loss = loss_cls(fake_pred, fake_label, ignore_index=10)
    torch_loss = torch.nn.functional.cross_entropy(
        fake_pred,
        fake_label,
        ignore_index=10,
        reduction='sum',
        weight=class_weight) / 12.0
    assert torch.allclose(loss, torch_loss)

    # test bce loss
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        class_weight=None,
        loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)

    fake_pred = torch.randn(2, 10).float()
    fake_label = torch.rand(2, 10).float()
    loss = loss_cls(fake_pred, fake_label)
    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_pred, fake_label, reduction='mean')
    assert torch.allclose(loss, torch_loss)

    fake_label[0, [1, 2, 5, 7]] = -1  # set ignore_index
    fake_label[1, [0, 5, 8, 9]] = -1
    loss = loss_cls(fake_pred, fake_label, ignore_index=-1)
    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_pred[fake_label != -1],
        fake_label[fake_label != -1],
        reduction='mean')
    assert torch.allclose(loss, torch_loss)

    # test ignore index and weight
    weight = torch.rand(2, 10)
    loss = loss_cls(fake_pred, fake_label, weight=weight, ignore_index=-1)
    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_pred[fake_label != -1],
        fake_label[fake_label != -1],
        reduction='mean',
        weight=weight[fake_label != -1])
    assert torch.allclose(loss, torch_loss)

    # TODO test use_mask
