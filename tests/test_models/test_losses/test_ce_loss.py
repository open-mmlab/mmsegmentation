# Copyright (c) OpenMMLab. All rights reserved.
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
        loss_weight=1.0,
        loss_name='loss_ce')
    loss_cls = build_loss(loss_cls_cfg)
    assert loss_cls.extra_repr() == 'avg_non_ignore=False'
    fake_pred = torch.Tensor([[100, -100]])
    fake_label = torch.Tensor([1]).long()
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(40.))

    # test loss with class weights, reduction='none'
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=[0.8, 0.2],
        loss_weight=1.0,
        reduction='none',
        loss_name='loss_ce')
    loss_cls = build_loss(loss_cls_cfg)
    loss_cls.extra_repr()
    fake_pred = torch.Tensor([[100, -100]])
    fake_label = torch.Tensor([1]).long()
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(40.))

    # test loss with class weights, reduction='sum'
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=[0.8, 0.2],
        loss_weight=1.0,
        reduction='sum',
        loss_name='loss_ce')
    loss_cls = build_loss(loss_cls_cfg)
    loss_cls.extra_repr()
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

    loss_cls_cfg = dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(100.))

    # test results in different situation
    # such as `avg_non_ignore`, `reduction` in ce and bce loss.
    fake_pred = torch.full(size=(2, 21, 8, 8), fill_value=0.5)
    fake_label = torch.ones(2, 8, 8).long()
    assert torch.allclose(
        loss_cls(fake_pred, fake_label), torch.tensor(0.9503), atol=1e-4)

    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls = build_loss(loss_cls_cfg)
    fake_label[:, 0, 0] = 255
    assert torch.allclose(
        loss_cls(fake_pred, fake_label, ignore_index=255),
        torch.tensor(0.9503),
        atol=1e-4)

    # test avg_non_ignore=False for bce
    # in this case, `avg_factor` would not be calculated
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        loss_weight=1.0,
        avg_non_ignore=False)
    loss_cls = build_loss(loss_cls_cfg)
    fake_label[:, 0, 0] = 255
    assert torch.allclose(
        loss_cls(fake_pred, fake_label, ignore_index=255),
        torch.tensor(0.9354),
        atol=1e-4)

    # test avg_non_ignore would not affect bce/ce
    # when reduction='sum'/'mean'/'none'
    fake_pred = torch.full(size=(2, 21, 8, 8), fill_value=0.5)
    fake_label = torch.ones(2, 8, 8).long()
    fake_label[:, 0, 0] = 255

    # test avg_non_ignore would not affect bce
    # when reduction='sum'
    loss_cls_cfg1 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='sum',
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls1 = build_loss(loss_cls_cfg1)
    loss_cls_cfg2 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='sum',
        loss_weight=1.0,
        avg_non_ignore=False)
    loss_cls2 = build_loss(loss_cls_cfg2)
    assert torch.allclose(
        loss_cls1(fake_pred, fake_label, ignore_index=255) / fake_pred.numel(),
        loss_cls2(fake_pred, fake_label, ignore_index=255) / fake_pred.numel(),
        atol=1e-4)

    # test avg_non_ignore would not affect bce
    # when reduction='none'
    loss_cls_cfg1 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='none',
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls1 = build_loss(loss_cls_cfg1)
    loss_cls_cfg2 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='none',
        loss_weight=1.0,
        avg_non_ignore=False)
    loss_cls2 = build_loss(loss_cls_cfg2)
    assert torch.allclose(
        loss_cls1(fake_pred, fake_label, ignore_index=255).sum() /
        fake_pred.numel(),
        loss_cls2(fake_pred, fake_label, ignore_index=255).sum() /
        fake_pred.numel(),
        atol=1e-4)

    # test avg_non_ignore would not affect bce
    # when reduction='mean'
    loss_cls_cfg1 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls1 = build_loss(loss_cls_cfg1)
    loss_cls_cfg2 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        loss_weight=1.0,
        avg_non_ignore=False)
    loss_cls2 = build_loss(loss_cls_cfg2)
    assert torch.allclose(
        loss_cls1(fake_pred, fake_label, ignore_index=255).sum() /
        fake_pred.numel(),
        loss_cls2(fake_pred, fake_label, ignore_index=255).sum() /
        fake_pred.numel(),
        atol=1e-4)

    # test avg_non_ignore would not affect ce
    # when reduction='sum'
    loss_cls_cfg1 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='sum',
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls1 = build_loss(loss_cls_cfg1)
    loss_cls_cfg2 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='sum',
        loss_weight=1.0,
        avg_non_ignore=False)
    loss_cls2 = build_loss(loss_cls_cfg2)
    assert torch.allclose(
        loss_cls1(fake_pred, fake_label, ignore_index=255) / fake_pred.numel(),
        loss_cls2(fake_pred, fake_label, ignore_index=255) / fake_pred.numel(),
        atol=1e-4)

    # test avg_non_ignore would not affect ce
    # when reduction='none'
    loss_cls_cfg1 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='none',
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls1 = build_loss(loss_cls_cfg1)
    loss_cls_cfg2 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='none',
        loss_weight=1.0,
        avg_non_ignore=False)
    loss_cls2 = build_loss(loss_cls_cfg2)
    assert torch.allclose(
        loss_cls1(fake_pred, fake_label, ignore_index=255).sum() /
        fake_pred.numel(),
        loss_cls2(fake_pred, fake_label, ignore_index=255).sum() /
        fake_pred.numel(),
        atol=1e-4)

    # test avg_non_ignore would not affect ce
    # when reduction='mean'
    loss_cls_cfg1 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='mean',
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls1 = build_loss(loss_cls_cfg1)
    loss_cls_cfg2 = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='mean',
        loss_weight=1.0,
        avg_non_ignore=False)
    loss_cls2 = build_loss(loss_cls_cfg2)
    assert torch.allclose(
        loss_cls1(fake_pred, fake_label, ignore_index=255).sum() /
        fake_pred.numel(),
        loss_cls2(fake_pred, fake_label, ignore_index=255).sum() /
        fake_pred.numel(),
        atol=1e-4)

    # test cross entropy loss has name `loss_ce`
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0,
        loss_name='loss_ce')
    loss_cls = build_loss(loss_cls_cfg)
    assert loss_cls.loss_name == 'loss_ce'

    # test ce loss with ignore index
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='mean',
        class_weight=None,
        loss_weight=1.0,
        avg_non_ignore=True)
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

    # test ce loss with ignore index
    # avg_non_ignore=False
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='mean',
        class_weight=None,
        loss_weight=1.0,
        avg_non_ignore=False)
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
        fake_pred, fake_label, ignore_index=10,
        reduction='sum') / fake_label.numel()
    assert torch.allclose(loss, torch_loss)

    # test ignore index and class_weight
    class_weight = torch.rand(5)
    class_weight /= class_weight.sum()
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='mean',
        class_weight=class_weight,
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls = build_loss(loss_cls_cfg)
    loss = loss_cls(fake_pred, fake_label, ignore_index=10)
    torch_loss = torch.nn.functional.cross_entropy(
        fake_pred,
        fake_label,
        ignore_index=10,
        reduction='sum',
        weight=class_weight) / 12.0
    assert torch.allclose(loss, torch_loss)

    # test ignore index and class_weight
    # avg_non_ignore=False
    class_weight = torch.rand(5)
    class_weight /= class_weight.sum()
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='mean',
        class_weight=class_weight,
        loss_weight=1.0,
        avg_non_ignore=False)
    loss_cls = build_loss(loss_cls_cfg)
    loss = loss_cls(fake_pred, fake_label, ignore_index=10)
    torch_loss = torch.nn.functional.cross_entropy(
        fake_pred,
        fake_label,
        ignore_index=10,
        reduction='sum',
        weight=class_weight) / fake_label.numel()
    assert torch.allclose(loss, torch_loss)

    # test ce loss
    # without ignore index and avg_non_ignore
    # doesn't affect the results
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='mean',
        class_weight=None,
        loss_weight=1.0,
        avg_non_ignore=False)
    loss_cls = build_loss(loss_cls_cfg)

    fake_pred = torch.randn(2, 5, 10).float()  # 5-way classification
    fake_label = torch.randint(0, 5, (2, 10)).long()
    loss = loss_cls(fake_pred, fake_label)
    torch_loss = torch.nn.functional.cross_entropy(
        fake_pred, fake_label, reduction='mean')
    assert torch.allclose(loss, torch_loss)

    torch_loss = torch.nn.functional.cross_entropy(
        fake_pred, fake_label, reduction='sum') / fake_label.numel()
    assert torch.allclose(loss, torch_loss)

    # test bce loss
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        class_weight=None,
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls = build_loss(loss_cls_cfg)

    fake_pred = torch.randn(2, 10).float()
    fake_label = torch.rand(2, 10).float()
    loss = loss_cls(fake_pred, fake_label)
    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_pred, fake_label, reduction='mean')
    assert torch.allclose(loss, torch_loss)

    fake_label[0, [1, 2, 5, 7]] = 10  # set ignore_index
    fake_label[1, [0, 5, 8, 9]] = 10
    loss = loss_cls(fake_pred, fake_label, ignore_index=10)
    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_pred[fake_label != 10],
        fake_label[fake_label != 10],
        reduction='mean')
    assert torch.allclose(loss, torch_loss)

    # test bce loss
    # without ignore index and avg_non_ignore
    # doesn't affect the results
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        class_weight=None,
        loss_weight=1.0,
        avg_non_ignore=False)
    loss_cls = build_loss(loss_cls_cfg)

    fake_pred = torch.randn(2, 10).float()
    fake_label = torch.rand(2, 10).float()
    loss = loss_cls(fake_pred, fake_label)
    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_pred, fake_label, reduction='mean')
    assert torch.allclose(loss, torch_loss)

    # test bce loss
    # with ignore index and avg_non_ignore=False
    # doesn't affect the results
    fake_label[0, [1, 2, 5, 7]] = 10  # set ignore_index
    fake_label[1, [0, 5, 8, 9]] = 10
    loss = loss_cls(fake_pred, fake_label, ignore_index=10)
    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_pred[fake_label != 10],
        fake_label[fake_label != 10],
        reduction='sum') / fake_label.numel()
    assert torch.allclose(loss, torch_loss)

    fake_pred = torch.randn(2, 10).float()
    fake_label = torch.rand(2, 10).float()
    loss = loss_cls(fake_pred, fake_label)
    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_pred, fake_label, reduction='mean')
    assert torch.allclose(loss, torch_loss)

    # test bce
    # with ignore index and weight and avg_non_ignore=False
    weight = torch.rand(2, 10)
    loss = loss_cls(fake_pred, fake_label, weight=weight, ignore_index=10)
    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_pred[fake_label != 10],
        fake_label[fake_label != 10],
        reduction='mean',
        weight=weight[fake_label != 10])
    assert torch.allclose(loss, torch_loss)

    # test ce loss with ignore index and reduction='none'
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='none',
        class_weight=None,
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls = build_loss(loss_cls_cfg)

    fake_pred = torch.randn(2, 5, 10).float()  # 5-way classification
    fake_label = torch.randint(0, 5, (2, 10)).long()
    loss = loss_cls(fake_pred, fake_label)
    torch_loss = torch.nn.functional.cross_entropy(
        fake_pred, fake_label, reduction='none')
    assert torch.allclose(loss, torch_loss)

    # test ce loss with ignore index and reduction='sum'
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='sum',
        class_weight=None,
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls = build_loss(loss_cls_cfg)

    fake_pred = torch.randn(2, 5, 10).float()  # 5-way classification
    fake_label = torch.randint(0, 5, (2, 10)).long()
    loss = loss_cls(fake_pred, fake_label)
    torch_loss = torch.nn.functional.cross_entropy(
        fake_pred, fake_label, reduction='sum')
    assert torch.allclose(loss, torch_loss)

    # test ce loss with ignore index and reduction='none'
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        reduction='none',
        class_weight=None,
        loss_weight=1.0,
        avg_non_ignore=True)
    loss_cls = build_loss(loss_cls_cfg)

    fake_pred = torch.randn(2, 5, 10).float()  # 5-way classification
    fake_label = torch.randint(0, 5, (2, 10)).long()
    loss = loss_cls(fake_pred, fake_label)
    torch_loss = torch.nn.functional.cross_entropy(
        fake_pred, fake_label, reduction='none')
    assert torch.allclose(loss, torch_loss)

    # TODO test use_mask
