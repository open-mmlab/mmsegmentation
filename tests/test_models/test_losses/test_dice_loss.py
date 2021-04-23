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
    dice_loss(logits, labels)

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
