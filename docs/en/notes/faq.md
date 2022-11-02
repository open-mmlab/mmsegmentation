# Frequently Asked Questions (FAQ)

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmsegmentation/blob/master/.github/ISSUE_TEMPLATE/error-report.md/) and make sure you fill in all required information in the template.

## Installation

The compatible MMSegmentation and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

| MMSegmentation version |        MMCV version         | MMClassification version |
| :--------------------: | :-------------------------: | :----------------------: |
|        1.0.0rc1        |      mmcv >= 2.0.0rc1       |     mmcls>=1.0.0rc0      |
|        1.0.0rc0        |      mmcv >= 2.0.0rc1       |     mmcls>=1.0.0rc0      |
|         master         | mmcv-full>=1.4.4, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
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

## How to know the number of GPUs needed to train the model

- Infer from the name of the config file of the model. You can refer to the `Config Name Style` part of [Learn about Configs](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/tutorials/config.md). For example, for config file with name `segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py`, `8xb1` means training the model corresponding to it needs 8 GPUs, and the batch size of each GPU is 1.
- Infer from the log file. Open the log file of the model and search `nGPU` in the file. The number of figures following `nGPU` is the number of GPUs needed to train the model. For instance, searching for `nGPU` in the log file yields the record `nGPU 0,1,2,3,4,5,6,7`, which indicates that eight GPUs are needed to train the model.
