# 4. Training Tricks

MMSegmentation support following training tricks out of box.

## Different Learning Rate(LR) for Backbone and Heads

In semantic segmentation, some methods make the LR of heads larger than backbone to achieve better performance or faster convergence.

In MMSegmentation, you may add following lines to config to make the LR of heads 10 times of backbone.
```python
optimizer_config=dict(
    paramwise_cfg = dict(
        custom_keys={
            'head': dict(lr_mult=10.)}))
```
With this modification, the LR of any parameter group with `'head'` in name will be multiplied by 10.
You may refer to [MMCV doc](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.DefaultOptimizerConstructor) for further details.

## Online Hard Example Mining (OHEM)
We implement pixel sampler [here](https://github.com/open-mmlab/mmsegmentation/tree/master/mmseg/core/seg/sampler) for training sampling.
Here is an example config of training PSPNet with OHEM enabled.
```python
_base_ = './pspnet_r50-d8_512x1024_40k_cityscapes.py'
model=dict(
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)) )
```
In this way, only pixels with confidence score under 0.7 are used to train. And we keep at least 100000 pixels during training.
