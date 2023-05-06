# 训练技巧

MMSegmentation 支持如下训练技巧：

## 主干网络和解码头组件使用不同的学习率 (Learning Rate, LR)

在语义分割里，一些方法会让解码头组件的学习率大于主干网络的学习率，这样可以获得更好的表现或更快的收敛。

在 MMSegmentation 里面，您也可以在配置文件里添加如下行来让解码头组件的学习率是主干组件的10倍。

```python
optim_wrapper=dict(
    paramwise_cfg = dict(
        custom_keys={
            'head': dict(lr_mult=10.)}))
```

通过这种修改，任何被分组到 `'head'` 的参数的学习率都将乘以10。您也可以参照 [MMEngine 文档](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html#id6)  获取更详细的信息。

## 在线难样本挖掘 (Online Hard Example Mining, OHEM)

MMSegmentation 中实现了像素采样器，训练时可以对特定像素进行采样，例如 OHEM(Online Hard Example Mining)，可以解决样本不平衡问题，
如下例子是使用 PSPNet 训练并采用 OHEM 策略的配置：

```python
_base_ = './pspnet_r50-d8_512x1024_40k_cityscapes.py'
model=dict(
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)) )
```

通过这种方式，只有置信分数在0.7以下的像素值点会被拿来训练。在训练时我们至少要保留100000个像素值点。如果 `thresh` 并未被指定，前 `min_kept`
个损失的像素值点才会被选择。

## 类别平衡损失 (Class Balanced Loss)

对于不平衡类别分布的数据集，您也许可以改变每个类别的损失权重。这里以 cityscapes 数据集为例：

```python
_base_ = './pspnet_r50-d8_512x1024_40k_cityscapes.py'
model=dict(
    decode_head=dict(
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            # DeepLab 对 cityscapes 使用这种权重
            class_weight=[0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                        1.0865, 1.0955, 1.0865, 1.1529, 1.0507])))
```

`class_weight` 将被作为 `weight` 参数，传递给 `CrossEntropyLoss`。详细信息请参照 [PyTorch 文档](https://pytorch.org/docs/stable/nn.html?highlight=crossentropy#torch.nn.CrossEntropyLoss) 。

## 同时使用多种损失函数 (Multiple Losses)

对于训练时损失函数的计算，我们目前支持多个损失函数同时使用。 以 `unet` 使用 `DRIVE` 数据集训练为例，
使用 `CrossEntropyLoss` 和 `DiceLoss` 的 `1:3` 的加权和作为损失函数。配置文件写为:

```python
_base_ = './fcn_unet_s5-d16_64x64_40k_drive.py'
model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ]),
    auxiliary_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ]),
)
```

通过这种方式，确定训练过程中损失函数的权重 `loss_weight` 和在训练日志里的名字 `loss_name`。

注意： `loss_name` 的名字必须带有 `loss_` 前缀，这样它才能被包括在计算图里。
