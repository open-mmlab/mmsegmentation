# 将 MMSeg 模型调优及部署到 NVIDIA Jetson 平台教程
**本教程所用 mmsegmentation 版本：v1.1.1**  
**本教程所用 NVIDIA Jetson 设备：NVIDIA Jetson AGX Orin 64G**
<div align="center">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/b5466cfd-71a9-4e06-9823-c253a97d57b5" alt="Smiley face" width="50%">  
</div>

## 1 配置 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
* 根据[安装和验证](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/get_started.md)文档，完成开发 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 所需的 [`pytorch`](https://pytorch.org/get-started/locally/)、[`mmcv`](https://github.com/open-mmlab/mmcv)、[`mmengine`](https://github.com/open-mmlab/mmengine) 等环境依赖安装。
* 从 GitHub 使用 git clone 命令完成 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 下载。网络不好的同学，可通过 [MMSeg GitHub](https://github.com/open-mmlab/mmsegmentation) 页面进行 zip 的下载。
    ```bash
    git clone https://github.com/open-mmlab/mmsegmentation.git
    ```
* 使用 `pip install -v -e.` 命令动态安装 mmsegmentation 。
    ```bash 
    cd mmsegmentation
    pip install -v -e .
    ```
    提示成功安装后，可通过 `pip list` 命令查看到 mmsegmentation 已通过本地安装方式安装到了您的环境中。
    ![mmseg-install](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/a9c7bcc9-cdcc-40a4-bd7b-8153195549c8)

## 2 准备您的数据集
* 本教程使用遥感图像语义分割数据集 [potsdam](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/user_guides/2_dataset_prepare.md#isprs-potsdam) 作为示例。
* 根据 [potsdam 数据准备](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/user_guides/2_dataset_prepare.md#isprs-potsdam)文档，进行数据集下载及 MMSeg 格式的准备。
* 数据集介绍： potsdam 数据集是以德国一个典型的历史城市 Potsdam 命名的，该城市有着大建筑群、狭窄的街道和密集的建筑结构。 potsdam 数据集包含 38 幅 6000x6000 像素的图像，空间分辨率为 5cm，数据集的示例如下图：
    ![potsdam-img](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/3bc0a75b-1693-4ae6-aeea-ad502e955068)

## 3 从 config 页面下载对应的 pth 权重文件
这里以 [`deeplabv3plus_r101-d8_4xb4-80k_potsdam-512x512.py`](../../configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-80k_potsdam-512x512.py) 配置文件举例，在 [configs](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/deeplabv3plus#potsdam) 页面下载权重文件，
![pth](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/8f747362-caf4-406c-808d-4ca72babb209)

## 4 通过 OpenMMLab deployee 进行模型转换及测速
### 4.1 模型转换
在该部分中，OpenMMLab 官网提供了模型转换及模型测速的 GUI 界面，无需任何代码，即可通过确认对应选项完成模型 ONNX 格式`xxxx.onnx` 和 TensorRT `.engine`格式的转换。
如您的自定义 config 文件中有相对引用关系，如
```python
# xxxx.py
_base_ = ['../_base_/models/']
```
请使用以下代码消除相对引用关系，以上传完整的 config
```python 
import mmengine

mmengine.Config.fromfile("configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-80k_potsdam-512x512.py").dump("My_config.py")
```
使用上述代码后，您能够看到，在`My_config.py`包含着完整的配置文件，无相对引用。这时，上传模型 config 至网页内。

### 4.2 模型测速

在完成模型转换后可通过**模型测速**界面，完成在真实设备上的模型的测速。

## 5 通过 OpenMMLab mmdeploy repo 进行模型转换
该部分可以通过 mmdeploy 库对 mmseg 训练好的模型进行推理格式的转换。这里给出一个示例，具体文档可见[ mmdeploy 模型转换文档]()。
### 5.1 通过源码构建 mmdeploy 库
在您安装 mmsegmentation 库的虚拟环境下，通过 `git clone`命令从 GitHub 拉取 [mmdeploy]()
    ```bash
    git clone xxxxx
    ```
### 5.2 如您的config
