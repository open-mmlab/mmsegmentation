from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.models.backbones import HRNet


def test_hrnet_backbone():
    # Test HRNET with two stage frozen

    extra = dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(4, ),
            num_channels=(64, )),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='BASIC',
            num_blocks=(4, 4),
            num_channels=(32, 64)),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(32, 64, 128)),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(32, 64, 128, 256)))
    frozen_stages = 2
    model = HRNet(extra, frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    assert model.norm1.training is False

    for layer in [model.conv1, model.norm1]:
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        if i == 1:
            layer = getattr(model, f'layer{i}')
            transition = getattr(model, f'transition{i}')
        elif i == 4:
            layer = getattr(model, f'stage{i}')
        else:
            layer = getattr(model, f'stage{i}')
            transition = getattr(model, f'transition{i}')

        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

        for mod in transition.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in transition.parameters():
            assert param.requires_grad is False
