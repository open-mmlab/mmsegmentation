# 将 MMSeg 模型调优及部署到 NVIDIA Jetson 平台教程
请先查阅[MMSegmentation 模型部署](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/user_guides/5_deployment.md)文档
**本教程所用 mmsegmentation 版本：v1.1.2**  
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

## 3 从 config 页面下载模型的 pth 权重文件
这里以 [`deeplabv3plus_r101-d8_4xb4-80k_potsdam-512x512.py`](../../configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-80k_potsdam-512x512.py) 配置文件举例，在 [configs](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/deeplabv3plus#potsdam) 页面下载权重文件，
![pth](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/8f747362-caf4-406c-808d-4ca72babb209)

## 4 通过 [OpenMMLab deployee](https://platform.openmmlab.com/deploee) 进行模型转换及测速
### 4.1 模型转换
在该部分中，[OpenMMLab 官网](https://platform.openmmlab.com/deploee)提供了模型转换及模型测速的交互界面，无需任何代码，即可通过选择对应选项完成模型 ONNX 格式`xxxx.onnx` 和 TensorRT `.engine`格式的转换。
如您的自定义 config 文件中有相对引用关系，如：
```python
# xxxx.py
_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/potsdam.py',        
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
```
您可以使用以下代码消除相对引用关系，以生成完整的 config 文件。
```python 
import mmengine

mmengine.Config.fromfile("configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-80k_potsdam-512x512.py").dump("My_config.py")
```
使用上述代码后，您能够看到，在`My_config.py`包含着完整的配置文件，无相对引用。这时，上传模型 config 至网页内对应处。  
#### 4.1.1 新建转换任务
<div align="center">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/21c33be0-378e-43c0-869e-3d1afeb7d817" alt="模型转换界面" width="50%">  
</div>

### 4.2 模型测速

在完成模型转换后可通过**模型测速**界面，完成在真实设备上的模型的测速。

## 5 通过 OpenMMLab mmdeploy repo 进行模型转换
该部分可以通过 mmdeploy 库对 mmseg 训练好的模型进行推理格式的转换。这里给出一个示例，具体文档可见[ mmdeploy 模型转换文档]()。
### 5.1 通过源码构建 mmdeploy 库
在您安装 mmsegmentation 库的虚拟环境下，通过 `git clone`命令从 GitHub 拉取 [mmdeploy]()
    ```bash
    git clone xxxxx
    ```
### 5.2 模型转换
如您的 config 中含有相对引用，仍需进行消除，如[4.1 模型转换](#4.1-模型转换)所述

# 6 Jetson 环境准备
参考[如何在 Jetson 模组上安装 MMDeploy](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/01-how-to-build/jetsons.md)文档，完成在 Jetson 上的环境准备工作。  
**注**：安装 Pytorch，可查阅 [NVIDIA Jetson Pytorch 安装文档](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/01-how-to-build/jetsons.md)安装最新的 Pytorch。
## 6.1 创建虚拟环境
```bash
conda create -n {您虚拟环境的名字} python={python版本}
``` 
## 6.2 虚拟环境内安装Pytorch
```bash
sudo apt-get -y update; 
sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev;
```
```bash
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
```

```bash
python3 -m pip install --upgrade pip; python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3' export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"; python3 -m pip install --upgrade protobuf; python3 -m pip install --no-cache $TORCH_INSTALL
```
执行以上命令后，您将能看到以下提示，即为安装成功。
```bash
Successfully installed MarkupSafe-2.1.3 filelock-3.12.2 jinja2-3.1.2 mpmath-1.3.0 networkx-3.1 sympy-1.12 torch-2.0.0+nv23.5
```


## 6.3 将 Jetson Pack 自带的 tensorrt 拷贝至虚拟环境下
请参考[配置 TensorRT](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/01-how-to-build/jetsons.md#%E9%85%8D%E7%BD%AE-tensorrt)。  
JetPack SDK 自带 TensorRT。 但是为了能够在 Conda 环境中成功导入，我们需要将 TensorRT 拷贝进先前创建的 Conda 环境中。
```bash
export PYTHON_VERSION=`python3 --version | cut -d' ' -f 2 | cut -d'.' -f1,2`
cp -r /usr/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* ~/miniconda/envs/{您的虚拟环境名字}/lib/python${PYTHON_VERSION}/site-packages/
```
## 6.4 安装 MMCV
MMCV 还未提供针对 Jetson 平台的预编译包，因此我们需要从源对其进行编译。
```bash
sudo apt-get install -y libssl-dev
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -e .
```
## 安装 ONNX
不要安装最新的 ONNX，推荐的 ONNX 版本是 1.10.0。  
<font color="red">注：以下方式二选一</font>  
* conda
    ```bash
    conda install -c conda-forge onnx
    ```
* pip
    ```bash
    python3 -m pip install onnx==1.10.0
    ```
