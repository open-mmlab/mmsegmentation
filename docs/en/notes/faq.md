# Frequently Asked Questions (FAQ)

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmsegmentation/blob/master/.github/ISSUE_TEMPLATE/error-report.md/) and make sure you fill in all required information in the template.

## Installation

The compatible MMSegmentation, MMCV and MMEngine versions are as below. Please install the correct versions of them to avoid installation issues.

| MMSegmentation version |          MMCV version          | MMEngine version  | MMClassification (optional) version | MMDetection (optional) version |
| :--------------------: | :----------------------------: | :---------------: | :---------------------------------: | :----------------------------: |
|     dev-1.x branch     |        mmcv >= 2.0.0rc4        | MMEngine >= 0.2.0 |           mmcls>=1.0.0rc0           |         mmdet>3.0.0rc5         |
|       1.x branch       |        mmcv >= 2.0.0rc4        | MMEngine >= 0.2.0 |           mmcls>=1.0.0rc0           |         mmdet>3.0.0rc5         |
|        1.0.0rc5        |        mmcv >= 2.0.0rc4        | MMEngine >= 0.2.0 |           mmcls>=1.0.0rc0           |         mmdet>3.0.0rc5         |
|        1.0.0rc4        |        mmcv == 2.0.0rc3        | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |  mmdet>=3.0.0rc4, \<=3.0.0rc5  |
|        1.0.0rc3        |        mmcv == 2.0.0rc3        | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |  mmdet>=3.0.0rc4  \<=3.0.0rc5  |
|        1.0.0rc2        |        mmcv == 2.0.0rc3        | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |  mmdet>=3.0.0rc4  \<=3.0.0rc5  |
|        1.0.0rc1        | mmcv >= 2.0.0rc1, \<=2.0.0rc3> | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |          Not required          |
|        1.0.0rc0        | mmcv >= 2.0.0rc1, \<=2.0.0rc3> | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |          Not required          |

Notes: To install MMSegmentation 0.x and master branch, please refer to [the faq 0.x document](https://mmsegmentation.readthedocs.io/en/latest/faq.html#installation) to check compatible versions of MMCV.

## How to know the number of GPUs needed to train the model

- Infer from the name of the config file of the model. You can refer to the `Config Name Style` part of [Learn about Configs](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/tutorials/config.md). For example, for config file with name `segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py`, `8xb1` means training the model corresponding to it needs 8 GPUs, and the batch size of each GPU is 1.
- Infer from the log file. Open the log file of the model and search `nGPU` in the file. The number of figures following `nGPU` is the number of GPUs needed to train the model. For instance, searching for `nGPU` in the log file yields the record `nGPU 0,1,2,3,4,5,6,7`, which indicates that eight GPUs are needed to train the model.
