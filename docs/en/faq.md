# Frequently Asked Questions (FAQ)

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmsegmentation/blob/master/.github/ISSUE_TEMPLATE/error-report.md/) and make sure you fill in all required information in the template.

## Installation

The compatible MMSegmentation and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

| MMSegmentation version |        MMCV version         | MMClassification version |
| :--------------------: | :-------------------------: | :----------------------: |
|         master         |  mmcv-full>=1.5.0, \<1.7.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.29.0         |  mmcv-full>=1.5.0, \<1.7.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.28.0         |  mmcv-full>=1.5.0, \<1.7.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.27.0         |  mmcv-full>=1.5.0, \<1.7.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.26.0         | mmcv-full>=1.5.0, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.25.0         | mmcv-full>=1.5.0, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.24.1         | mmcv-full>=1.4.4, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.23.0         | mmcv-full>=1.4.4, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.22.0         | mmcv-full>=1.4.4, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.21.1         | mmcv-full>=1.4.4, \<=1.6.0  |       Not required       |
|         0.20.2         | mmcv-full>=1.3.13, \<=1.6.0 |       Not required       |
|         0.19.0         | mmcv-full>=1.3.13, \<1.3.17 |       Not required       |
|         0.18.0         | mmcv-full>=1.3.13, \<1.3.17 |       Not required       |
|         0.17.0         | mmcv-full>=1.3.7, \<1.3.17  |       Not required       |
|         0.16.0         | mmcv-full>=1.3.7, \<1.3.17  |       Not required       |
|         0.15.0         | mmcv-full>=1.3.7, \<1.3.17  |       Not required       |
|         0.14.1         | mmcv-full>=1.3.7, \<1.3.17  |       Not required       |
|         0.14.0         |  mmcv-full>=1.3.1, \<1.3.2  |       Not required       |
|         0.13.0         |  mmcv-full>=1.3.1, \<1.3.2  |       Not required       |
|         0.12.0         |  mmcv-full>=1.1.4, \<1.3.2  |       Not required       |
|         0.11.0         |  mmcv-full>=1.1.4, \<1.3.0  |       Not required       |
|         0.10.0         |  mmcv-full>=1.1.4, \<1.3.0  |       Not required       |
|         0.9.0          |  mmcv-full>=1.1.4, \<1.3.0  |       Not required       |
|         0.8.0          |  mmcv-full>=1.1.4, \<1.2.0  |       Not required       |
|         0.7.0          |  mmcv-full>=1.1.2, \<1.2.0  |       Not required       |
|         0.6.0          |  mmcv-full>=1.1.2, \<1.2.0  |       Not required       |

You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

- "No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'".

  1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`.
  2. Install mmcv-full following the [installation instruction](get_started#best-practices).

## How to know the number of GPUs needed to train the model

- Infer from the name of the config file of the model. You can refer to the `Config Name Style` part of [Learn about Configs](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/tutorials/config.md). For example, for config file with name `segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py`, `8x1` means training the model corresponding to it needs 8 GPUs, and the batch size of each GPU is 1.
- Infer from the log file. Open the log file of the model and search `nGPU` in the file. The number of figures following `nGPU` is the number of GPUs needed to train the model. For instance, searching for `nGPU` in the log file yields the record `nGPU 0,1,2,3,4,5,6,7`, which indicates that eight GPUs are needed to train the model.

## What does the auxiliary head mean

Briefly, it is a deep supervision trick to improve the accuracy. In the training phase, `decode_head` is for decoding semantic segmentation output, `auxiliary_head` is just adding an auxiliary loss, the segmentation result produced by it has no impact to your model's result, it just works in training. You may read this [paper](https://arxiv.org/pdf/1612.01105.pdf) for more information.

## Why is the log file not created

In the train script, we call `get_root_logger`at Line 167, and `get_root_logger` in mmseg calls `get_logger` in mmcv, mmcv will return the same logger which has been initialized in 'mmsegmentation/tools/train.py' with the parameter `log_file`. There is only one logger (initialized with `log_file`) during training.
Ref: [https://github.com/open-mmlab/mmcv/blob/21bada32560c7ed7b15b017dc763d862789e29a8/mmcv/utils/logging.py#L9-L16](https://github.com/open-mmlab/mmcv/blob/21bada32560c7ed7b15b017dc763d862789e29a8/mmcv/utils/logging.py#L9-L16)

If you find the log file not been created, you might check if `mmcv.utils.get_logger` is called elsewhere.

## How to output the image for painting the segmentation mask when running the test script

In the test script, we provide `show-dir` argument to control whether output the painted images. Users might run the following command:

```shell
python tools/test.py {config} {checkpoint} --show-dir {/path/to/save/image} --opacity 1
```

## How to handle binary segmentation task

MMSegmentation uses `num_classes` and `out_channels` to control output of last layer `self.conv_seg` (More details could be found [here](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/decode_head.py).):

```python
def __init__(self,
             ...,
             ):
  ...
  if out_channels is None:
      if num_classes == 2:
          warnings.warn('For binary segmentation, we suggest using'
                        '`out_channels = 1` to define the output'
                        'channels of segmentor, and use `threshold`'
                        'to convert seg_logist into a prediction'
                        'applying a threshold')
      out_channels = num_classes

  if out_channels != num_classes and out_channels != 1:
      raise ValueError(
          'out_channels should be equal to num_classes,'
          'except binary segmentation set out_channels == 1 and'
          f'num_classes == 2, but got out_channels={out_channels}'
          f'and num_classes={num_classes}')

  if out_channels == 1 and threshold is None:
      threshold = 0.3
      warnings.warn('threshold is not defined for binary, and defaults'
                    'to 0.3')
  self.num_classes = num_classes
  self.out_channels = out_channels
  self.threshold = threshold
  ...
  self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
```

There are two types of calculating binary segmentation methods:

```python
...
if self.out_channels == 1:
    seg_logit = F.sigmoid(seg_logit)
else:
    seg_logit = F.softmax(seg_logit, dim=1)

...

if self.out_channels == 1:
    seg_pred = (seg_logit >
                self.decode_head.threshold).to(seg_logit).squeeze(1)
else:
    seg_pred = seg_logit.argmax(dim=1)
```

- When `out_channels=2`, using Cross Entropy Loss in training, using `F.softmax()` and `argmax()` to get prediction of each pixel in inference.

- When `out_channels=1`, we provide a parameter `threshold(default to 0.3)` in [#2016](https://github.com/open-mmlab/mmsegmentation/pull/2016), using Binary Cross Entropy Loss in training, using `F.sigmoid()` and `threshold` to get prediction of each pixel in inference.

More details about calculating segmentation prediction could be found in [encoder_decoder.py](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/segmentors/encoder_decoder.py):

In summary, to implement binary segmentation methods users should modify below parameters in config files:

- (1) `num_classes=2`, `out_channels=2`  and `use_sigmoid=False` in `CrossEntropyLoss`.

- (2) `num_classes=2`, `out_channels=1` and `use_sigmoid=True` in `CrossEntropyLoss`.

When taking solution (2), below is a modification example of [pspnet_unet_s5-d16.py](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/models/pspnet_unet_s5-d16.py):

```python
decode_head=dict(
    type='PSPHead',
    in_channels=64,
    in_index=4,
    num_classes=2,
    out_channels=1,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
auxiliary_head=dict(
    type='FCNHead',
    in_channels=128,
    in_index=3,
    num_classes=2,
    out_channels=1,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
```

## What does `reduce_zero_label` work for?

When [loading annotation](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/pipelines/loading.py#L91) in MMSegmentation, `reduce_zero_label (bool)` is provided to determine whether reduce all label value by 1:

```python
if self.reduce_zero_label:
    # avoid using underflow conversion
    gt_semantic_seg[gt_semantic_seg == 0] = 255
    gt_semantic_seg = gt_semantic_seg - 1
    gt_semantic_seg[gt_semantic_seg == 254] = 255
```

`reduce_zero_label` is usually used for datasets where 0 is background label, if `reduce_zero_label=True`, the pixels whose corresponding label is 0 would not be involved in loss calculation.
Noted that in binary segmentation task it is unnecessary to use `reduce_zero_label=True`, please take solutions we mentioned above.
