import torch
import torch.nn as nn
from mmcv.runner import build_optimizer_constructor


class SubModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=1, groups=2)
        self.gn = nn.GroupNorm(2, 2)
        self.param1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.decode_head = SubModel()
        self.auxiliary_head = SubModel()

    def forward(self, x):
        return x


base_lr = 0.01
base_wd = 0.0001
momentum = 0.9


def check_optimizer(optimizer,
                    model,
                    bias_lr_mult=1,
                    bias_decay_mult=1,
                    norm_decay_mult=1,
                    dwconv_decay_mult=1,
                    head_lr_mult=1):
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    model_parameters = list(model.parameters())
    assert len(param_groups) == len(model_parameters)
    for i, param in enumerate(model_parameters):
        param_group = param_groups[i]
        assert torch.equal(param_group['params'][0], param)
        assert param_group['momentum'] == momentum
    # param1
    param1 = param_groups[0]
    assert param1['lr'] == base_lr
    assert param1['weight_decay'] == base_wd
    # conv1.weight
    conv1_weight = param_groups[1]
    assert conv1_weight['lr'] == base_lr
    assert conv1_weight['weight_decay'] == base_wd
    # conv2.weight
    conv2_weight = param_groups[2]
    assert conv2_weight['lr'] == base_lr
    assert conv2_weight['weight_decay'] == base_wd
    # conv2.bias
    conv2_bias = param_groups[3]
    assert conv2_bias['lr'] == base_lr * bias_lr_mult
    assert conv2_bias['weight_decay'] == base_wd * bias_decay_mult
    # bn.weight
    bn_weight = param_groups[4]
    assert bn_weight['lr'] == base_lr
    assert bn_weight['weight_decay'] == base_wd * norm_decay_mult
    # bn.bias
    bn_bias = param_groups[5]
    assert bn_bias['lr'] == base_lr
    assert bn_bias['weight_decay'] == base_wd * norm_decay_mult
    # decode_head.param1
    sub_param1 = param_groups[6]
    assert sub_param1['lr'] == base_lr * head_lr_mult
    assert sub_param1['weight_decay'] == base_wd
    # decode_head.conv1.weight
    sub_conv1_weight = param_groups[7]
    assert sub_conv1_weight['lr'] == base_lr * head_lr_mult
    assert sub_conv1_weight['weight_decay'] == base_wd * dwconv_decay_mult
    # decode_head.conv1.bias
    sub_conv1_bias = param_groups[8]
    assert sub_conv1_bias['lr'] == base_lr * bias_lr_mult * head_lr_mult
    assert sub_conv1_bias['weight_decay'] == base_wd * dwconv_decay_mult
    # decode_head.gn.weight
    sub_gn_weight = param_groups[9]
    assert sub_gn_weight['lr'] == base_lr * head_lr_mult
    assert sub_gn_weight['weight_decay'] == base_wd * norm_decay_mult
    # decode_head.gn.bias
    sub_gn_bias = param_groups[10]
    assert sub_gn_bias['lr'] == base_lr * head_lr_mult
    assert sub_gn_bias['weight_decay'] == base_wd * norm_decay_mult
    # auxiliary_head.param1
    sub_param1 = param_groups[11]
    assert sub_param1['lr'] == base_lr * head_lr_mult
    assert sub_param1['weight_decay'] == base_wd
    # auxiliary_head.conv1.weight
    sub_conv1_weight = param_groups[12]
    assert sub_conv1_weight['lr'] == base_lr * head_lr_mult
    assert sub_conv1_weight['weight_decay'] == base_wd * dwconv_decay_mult
    # auxiliary_head.conv1.bias
    sub_conv1_bias = param_groups[13]
    assert sub_conv1_bias['lr'] == base_lr * bias_lr_mult * head_lr_mult
    assert sub_conv1_bias['weight_decay'] == base_wd * dwconv_decay_mult
    # auxiliary_head.gn.weight
    sub_gn_weight = param_groups[14]
    assert sub_gn_weight['lr'] == base_lr * head_lr_mult
    assert sub_gn_weight['weight_decay'] == base_wd * norm_decay_mult
    # auxiliary_head.gn.bias
    sub_gn_bias = param_groups[15]
    assert sub_gn_bias['lr'] == base_lr * head_lr_mult
    assert sub_gn_bias['weight_decay'] == base_wd * norm_decay_mult


def test_head_optimizer_constructor():
    model = ExampleModel()
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    paramwise_cfg = dict(
        bias_lr_mult=2,
        bias_decay_mult=0.5,
        norm_decay_mult=0,
        dwconv_decay_mult=0.1,
        head_lr_mult=10.)

    from mmseg.core.optimizer import HeadOptimizerConstructor  # noqa: F401
    optim_constructor_cfg = dict(
        type='HeadOptimizerConstructor',
        optimizer_cfg=optimizer_cfg,
        paramwise_cfg=paramwise_cfg)
    optim_constructor = build_optimizer_constructor(optim_constructor_cfg)
    optimizer = optim_constructor(model)
    check_optimizer(optimizer, model, **paramwise_cfg)
