# Frequently Asked Questions (FAQ)

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmsegmentation/blob/master/.github/ISSUE_TEMPLATE/error-report.md/) and make sure you fill in all required information in the template.

## Installation

The compatible MMSegmentation and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

| MMSegmentation version |        MMCV version         | MMClassification version |
| :--------------------: | :-------------------------: | :----------------------: |
|         master         | mmcv-full>=1.5.0, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
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

In the train script, we call `get_root_logger`at Line 167, and `get_root_logger` in mmseg calls `get_logger` in mmcv, mmcv will return the same logger which has beed initialized in 'mmsegmentation/tools/train.py' with the parameter `log_file`. There is only one logger (initialized with `log_file`) during training.
Ref: [https://github.com/open-mmlab/mmcv/blob/21bada32560c7ed7b15b017dc763d862789e29a8/mmcv/utils/logging.py#L9-L16](https://github.com/open-mmlab/mmcv/blob/21bada32560c7ed7b15b017dc763d862789e29a8/mmcv/utils/logging.py#L9-L16)

If you find the log file not been created, you might check if `mmcv.utils.get_logger` is called elsewhere.

## How to output the image for painting the segmentation mask when running the test script

In the test script, we provide `show-dir` argument to control whether output the painted images. Users might run the following command:

```shell
python tools/test.py {config} {checkpoint} --show-dir {/path/to/save/image} --opacity 1
```
