# Frequently Asked Questions (FAQ)

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/.github/ISSUE_TEMPLATE/error-report.md/) and make sure you fill in all required information in the template.

## Installation

The compatible MMSegmentation, MMCV and MMEngine versions are as below. Please install the correct versions of them to avoid installation issues.

| MMSegmentation version |          MMCV version          | MMEngine version  | MMClassification (optional) version | MMDetection (optional) version |
| :--------------------: | :----------------------------: | :---------------: | :---------------------------------: | :----------------------------: |
|     dev-1.x branch     |        mmcv >= 2.0.0rc4        | MMEngine >= 0.7.1 |           mmcls==1.0.0rc6           |         mmdet >= 3.0.0         |
|      main branch       |        mmcv >= 2.0.0rc4        | MMEngine >= 0.7.1 |           mmcls==1.0.0rc6           |         mmdet >= 3.0.0         |
|         1.0.0          |        mmcv >= 2.0.0rc4        | MMEngine >= 0.7.1 |           mmcls==1.0.0rc6           |         mmdet >= 3.0.0         |
|        1.0.0rc6        |        mmcv >= 2.0.0rc4        | MMEngine >= 0.5.0 |           mmcls>=1.0.0rc0           |       mmdet >= 3.0.0rc6        |
|        1.0.0rc5        |        mmcv >= 2.0.0rc4        | MMEngine >= 0.2.0 |           mmcls>=1.0.0rc0           |        mmdet>=3.0.0rc6         |
|        1.0.0rc4        |        mmcv == 2.0.0rc3        | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |  mmdet>=3.0.0rc4, \<=3.0.0rc5  |
|        1.0.0rc3        |        mmcv == 2.0.0rc3        | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |  mmdet>=3.0.0rc4, \<=3.0.0rc5  |
|        1.0.0rc2        |        mmcv == 2.0.0rc3        | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |  mmdet>=3.0.0rc4, \<=3.0.0rc5  |
|        1.0.0rc1        | mmcv >= 2.0.0rc1, \<=2.0.0rc3> | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |          Not required          |
|        1.0.0rc0        | mmcv >= 2.0.0rc1, \<=2.0.0rc3> | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |          Not required          |

Notes:

- MMClassification and MMDetatction are optional for MMSegmentation. If you didn't install them, `ConvNeXt` (required MMClassification) and MaskFormer, Mask2Former (required MMDetection) cannot be used. We recommend to install them with source code. Please refer to [MMClasssication](https://github.com/open-mmlab/mmclassification) and [MMDetection](https://github.com/open-mmlab/mmdetection) for more details about their installation.

- To install MMSegmentation 0.x and master branch, please refer to [the faq 0.x document](https://mmsegmentation.readthedocs.io/en/latest/faq.html#installation) to check compatible versions of MMCV.

- If you have installed an incompatible version of mmcv, please run `pip uninstall mmcv` to uninstall the installed mmcv first. If you have previously installed mmcv-full (which exists in OpenMMLab 1.x), please run `pip uninstall mmcv-full` to uninstall it.

- If "No module named 'mmcv'" appears, please follow the steps below;

  1. Use `pip uninstall mmcv` to uninstall the existing mmcv in the environment.
  2. Install the corresponding mmcv according to the [installation instructions](https://mmsegmentation.readthedocs.io/en/dev-1.x/get_started.html#best-practices).

## How to know the number of GPUs needed to train the model

- Infer from the name of the config file of the model. You can refer to the `Config Name Style` part of [Learn about Configs](../user_guides/1_config.md). For example, for config file with name `segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py`, `8xb1` means training the model corresponding to it needs 8 GPUs, and the batch size of each GPU is 1.
- Infer from the log file. Open the log file of the model and search `nGPU` in the file. The number of figures following `nGPU` is the number of GPUs needed to train the model. For instance, searching for `nGPU` in the log file yields the record `nGPU 0,1,2,3,4,5,6,7`, which indicates that eight GPUs are needed to train the model.

## What does the auxiliary head mean

Briefly, it is a deep supervision trick to improve the accuracy. In the training phase, `decode_head` is for decoding semantic segmentation output, `auxiliary_head` is just adding an auxiliary loss, the segmentation result produced by it has no impact to your model's result, it just works in training. You may read this [paper](https://arxiv.org/pdf/1612.01105.pdf) for more information.

## How to output the segmentation mask image when running the test script

In the test script, we provide `--out` argument to control whether output the painted images. Users might run the following command:

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${OUTPUT_DIR}
```

## How to handle binary segmentation task

MMSegmentation uses `num_classes` and `out_channels` to control output of last layer `self.conv_seg`. More details could be found [here](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/models/decode_heads/decode_head.py).

`num_classes` should be the same as number of types of labels, in binary segmentation task, dataset only has two types of labels: foreground and background, so `num_classes=2`. `out_channels` controls the output channel of last layer of model, it usually equals to `num_classes`.
But in binary segmentation task, there are two solutions:

- Set `out_channels=2`, using Cross Entropy Loss in training, using `F.softmax()` and `argmax()` to get prediction of each pixel in inference.

- Set `out_channels=1`, using Binary Cross Entropy Loss in training, using `F.sigmoid()` and `threshold` to get prediction of each pixel in inference. `threshold` is set 0.3 as default.

In summary, to implement binary segmentation methods users should modify below parameters in the `decode_head` and `auxiliary_head` configs. Here is a modification example of [pspnet_unet_s5-d16.py](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/models/pspnet_unet_s5-d16.py):

- (1) `num_classes=2`, `out_channels=2` and `use_sigmoid=False` in `CrossEntropyLoss`.

```python
decode_head=dict(
    type='PSPHead',
    in_channels=64,
    in_index=4,
    num_classes=2,
    out_channels=2,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
auxiliary_head=dict(
    type='FCNHead',
    in_channels=128,
    in_index=3,
    num_classes=2,
    out_channels=2,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
```

- (2) `num_classes=2`, `out_channels=1` and `use_sigmoid=True` in `CrossEntropyLoss`.

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

## Functionality of `reduce_zero_label`

The parameter type of `reduce_zero_label` in dataset is Boolean, which is default to False. It is used to ignore the dataset label 0. The specific method is to change label 0 to 255, and subtract 1 from the corresponding number of all the remaining labels. At the same time, set 255 as ignore index in the decode head, which means that it will not participate in the loss calculation.

Following is the specific implementation logic of `reduce_zero_label`:

```python
if self.reduce_zero_label:
    # avoid using underflow conversion
    gt_semantic_seg[gt_semantic_seg == 0] = 255
    gt_semantic_seg = gt_semantic_seg - 1
    gt_semantic_seg[gt_semantic_seg == 254] = 255
```

Whether your dataset needs to use `reduce_zero_label`, there are two types of situations:

- On [Potsdam](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/en/user_guides/2_dataset_prepare.md#isprs-potsdam) dataset, there are six classes: 0-Impervious surfaces, 1-Building, 2-Low vegetation, 3-Tree, 4-Car, 5-Clutter/background. However, this dataset provides two types of RGB labels, one with black pixels at the edges of the images, and the other without. For labels with black edges, in [dataset_converters.py](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/tools/dataset_converters/potsdam.py), it converts the black edges to label 0, and the other labels are 1-Impervious surfaces, 2-Building, 3-Low vegetation, 4-Tree, 5-Car, 6-Clutter/background. Therefore, in the dataset config [potsdam.py](https://github.com/open-mmlab/mmsegmentation/blob/ff95416c3b5ce8d62b9289f743531398efce534f/mmseg/datasets/potsdam.py#L23) `reduce_zero_label=True`。 If you are using labels without black edges, then there are only class 0-5 in the mask label. At this point, you should use `reduce_zero_label=False`. `reduce_zero_label` usage needs to be considered with your actual situation.
- On a dataset with class 0 as the background class, if you need to separate the background from the rest of your classes ultimately then you do not need to use `reduce_zero_label`, which in the dataset config settings should be `reduce_zero_label=False`

**Note:** Please confirm the number of original classes in the dataset. If there are only two classes, you should not use `reduce_zero_label` which is `reduce_zero_label=False`.
