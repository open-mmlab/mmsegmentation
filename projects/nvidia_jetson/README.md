# 将 MMSeg 模型调优及部署到 NVIDIA Jetson 平台教程

- 请先查阅[MMSegmentation 模型部署](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/user_guides/5_deployment.md)文档。
- **本教程所用 mmsegmentation 版本： v1.1.2**
- **本教程所用 NVIDIA Jetson 设备： NVIDIA Jetson AGX Orin 64G**

<div align="center">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/b5466cfd-71a9-4e06-9823-c253a97d57b5" alt="Smiley face" width="50%">
</div>

## 1 配置 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

- 根据[安装和验证](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/get_started.md)文档，完成开发 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 所需的 [`pytorch`](https://pytorch.org/get-started/locally/)、[`mmcv`](https://github.com/open-mmlab/mmcv)、[`mmengine`](https://github.com/open-mmlab/mmengine) 等环境依赖安装。
- 从 GitHub 使用 git clone 命令完成 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 下载。网络不好的同学，可通过 [MMSeg GitHub](https://github.com/open-mmlab/mmsegmentation) 页面进行 zip 的下载。
  ```bash
  git clone https://github.com/open-mmlab/mmsegmentation.git
  ```
- 使用 `pip install -v -e.` 命令动态安装 mmsegmentation 。
  ```bash
  cd mmsegmentation
  pip install -v -e .
  ```
  提示成功安装后，可通过 `pip list` 命令查看到 mmsegmentation 已通过本地安装方式安装到了您的环境中。
  ![mmseg-install](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/a9c7bcc9-cdcc-40a4-bd7b-8153195549c8)

## 2 准备您的数据集

- 本教程使用遥感图像语义分割数据集 [potsdam](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/user_guides/2_dataset_prepare.md#isprs-potsdam) 作为示例。
- 根据 [potsdam 数据准备](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/user_guides/2_dataset_prepare.md#isprs-potsdam)文档，进行数据集下载及 MMSeg 格式的准备。
- 数据集介绍： potsdam 数据集是以德国一个典型的历史城市 Potsdam 命名的，该城市有着大建筑群、狭窄的街道和密集的建筑结构。 potsdam 数据集包含 38 幅 6000x6000 像素的图像，空间分辨率为 5cm，数据集的示例如下图：
  ![potsdam-img](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/3bc0a75b-1693-4ae6-aeea-ad502e955068)

## 3 从 config 页面下载模型的 pth 权重文件

这里以 [`deeplabv3plus_r101-d8_4xb4-80k_potsdam-512x512.py`](../../configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-80k_potsdam-512x512.py) 配置文件举例，在 [configs](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/deeplabv3plus#potsdam) 页面下载权重文件，
![pth](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/8f747362-caf4-406c-808d-4ca72babb209)

## 4 通过 [OpenMMLab deployee](https://platform.openmmlab.com/deploee) 以交互式方式进行模型转换及测速

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

#### 创建转换任务

按照下图提示及自己的需求，创建转换任务并提交。

<div align="center">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/4918d2f9-d63c-480f-97f1-054529770cfd" alt="NVIDIA-Jetson" width="80%">
</div>

### 4.2 模型测速

在完成模型转换后可通过**模型测速**界面，完成在真实设备上的模型测速。

#### 创建测速任务

<div align="center">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/27340556-c81a-4ce3-8560-2c4727d3355e" alt="NVIDIA-Jetson" width="100%">
</div>

<div align="center">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/6f4fc3a9-ba9d-4829-8407-ed1470ba7bf3" alt="NVIDIA-Jetson" width="100%">
</div>

测速完成后，可在页面生成完整的测速报告。[查看测速报告示例](https://openmmlab-deploee.oss-cn-shanghai.aliyuncs.com/tmp/profile_speed/4352f5.txt)

## 5 通过 OpenMMLab mmdeploy 以命令行将模型转换为ONNX格式

该部分可以通过 mmdeploy 库对 mmseg 训练好的模型进行推理格式的转换。这里给出一个示例，具体文档可见[ mmdeploy 模型转换文档](../../docs/zh_cn/user_guides/5_deployment.md)。

### 5.1 通过源码构建 mmdeploy 库

在您安装 mmsegmentation 库的虚拟环境下，通过 `git clone`命令从 GitHub 克隆 [mmdeploy](https://github.com/open-mmlab/mmdeploy)

### 5.2 模型转换

如您的 config 中含有相对引用，仍需进行消除，如[4.1 模型转换](#4.1-模型转换)所述,
进入 mmdeploy 文件夹，执行以下命令，即可完成模型转换。

```bash
python tools/deploy.py \
    configs/mmseg/segmentation_onnxruntime_static-512x512.py \
    ../atl_config.py \
    ../deeplabv3plus_r18-d8_512x512_80k_potsdam_20211219_020601-75fd5bc3.pth \
    ../2_13_1024_5488_1536_6000.png \
    --work-dir ../atl_models \
    --device cpu \
    --show \
    --dump-info
```

```bash
# 使用方法
python ./tools/deploy.py \
    ${部署配置文件路径} \
    ${模型配置文件路径} \
    ${模型权重路径} \
    ${输入图像路径} \
    --work-dir ${用来保存日志和模型文件路径} \
    --device ${cpu/cuda:0} \
    --show \    # 是否显示检测的结果
    --dump-info # 是否输出 SDK 信息

```

执行成功后，您将能够看到以下提示，即为转换成功。

```bash
10/08 17:40:44 - mmengine - INFO - visualize pytorch model success.
10/08 17:40:44 - mmengine - INFO - All process success.
```

<div align="center">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/b752ccf8-903f-4ad3-ad7c-74fc25cb89a5" alt="NVIDIA-Jetson" width="400">
</div>

# 6 在 Jetson 平台进行转换及部署

## 6.1 环境准备

参考[如何在 Jetson 模组上安装 MMDeploy](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/01-how-to-build/jetsons.md)文档，完成在 Jetson 上的环境准备工作。
**注**：安装 Pytorch，可查阅 [NVIDIA Jetson Pytorch 安装文档](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/01-how-to-build/jetsons.md)安装最新的 Pytorch。

### 6.1.1 创建虚拟环境

```bash
conda create -n {您虚拟环境的名字} python={python版本}
```

### 6.1.2 虚拟环境内安装Pytorch

<font color="red">注意：</font>这里不要安装最新的 pytorch 2.0，因为 pyTorch 1.11 是最后一个使用 USE_DISTRIBUTED 构建的wheel，否则会在用mmdeploy进行模型转换的时候提示`AttributeError: module 'torch.distributed' has no attribute 'ReduceOp'`的错误。参考以下链接：https://forums.developer.nvidia.com/t/module-torch-distributed-has-no-attribute-reduceop/256581/6
下载`torch-1.11.0-cp38-cp38-linux_aarch64.whl`并安装

```bash
pip install torch-1.11.0-cp38-cp38-linux_aarch64.whl
```

执行以上命令后，您将能看到以下提示，即为安装成功。

```bash
Processing ./torch-1.11.0-cp38-cp38-linux_aarch64.whl
Requirement already satisfied: typing-extensions in /home/sirs/miniconda3/envs/openmmlab/lib/python3.8/site-packages (from torch==1.11.0) (4.7.1)
Installing collected packages: torch
Successfully installed torch-1.11.0
```

### 6.1.3 将 Jetson Pack 自带的 tensorrt 拷贝至虚拟环境下

请参考[配置 TensorRT](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/01-how-to-build/jetsons.md#%E9%85%8D%E7%BD%AE-tensorrt)。
JetPack SDK 自带 TensorRT。 但是为了能够在 Conda 环境中成功导入，我们需要将 TensorRT 拷贝进先前创建的 Conda 环境中。

```bash
export PYTHON_VERSION=`python3 --version | cut -d' ' -f 2 | cut -d'.' -f1,2`
cp -r /usr/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* ~/miniconda/envs/{您的虚拟环境名字}/lib/python${PYTHON_VERSION}/site-packages/
```

### 6.1.4 安装 MMCV

通过`mim install mmcv`或从源码对其进行编译。

```bash
pip install openmim
mim install mmcv
```

或者从源码对其进行编译。

```bash
sudo apt-get install -y libssl-dev
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -e .
```

<font color="red">注：pytorch版本发生变动后，需要重新编译mmcv。</font>

### 6.1.5 安装 ONNX

<font color="red">注：以下方式二选一</font>

- conda
  ```bash
  conda install -c conda-forge onnx
  ```
- pip
  ```bash
  python3 -m pip install onnx
  ```

### 6.1.6 安装 ONNX Runtime

根据网页 [ONNX Runtime](https://elinux.org/Jetson_Zoo#ONNX_Runtime) 选择合适的ONNX Runtime版本进行下载安装。
示例：

```bash
# Install pip wheel
$ pip3 install onnxruntime_gpu-1.10.0-cp38-cp38-linux_aarch64.whl

```

## 6.2 在 Jetson AGX Orin 进行模型转换及推理

### 6.2.1 ONNX 模型转换

同[4.1 模型转换](#4.1-模型转换)相同，在 Jetson 平台下进入安装好的虚拟环境，以及mmdeploy 目录，进行模型ONNX转换。

```bash
python tools/deploy.py \
    configs/mmseg/segmentation_onnxruntime_static-512x512.py \
    ../atl_config.py \
    ../deeplabv3plus_r18-d8_512x512_80k_potsdam_20211219_020601-75fd5bc3.pth \
    ../2_13_3584_2560_4096_3072.png \
    --work-dir ../atl_models \
    --device cpu \
    --show \
    --dump-info

```

<font color="red">注：</font> 如果报错提示内容：

```none
AttributeError: module 'torch.distributed' has no attribute 'ReduceOp'
```

可参考以下链接进行解决：https://forums.developer.nvidia.com/t/module-torch-distributed-has-no-attribute-reduceop/256581/6，即安装 pytorch 1.11.0 版本。

转换成功后，您将会看到如下信息以及包含 ONNX 模型的文件夹：

```bash
10/09 19:58:22 - mmengine - INFO - visualize pytorch model success.
10/09 19:58:22 - mmengine - INFO - All process success.
```

<div align="center">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/d68f1cf6-0e80-4261-91a3-6046b17de146" alt="NVIDIA-Jetson" width="400">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/70470a39-6a4f-4fd5-a06d-9b9d59a768ef" alt="NVIDIA-Jetson" width="160">
</div>

### 6.2.2 TensorRT 模型转换

更换部署trt配置文件，进行 TensorRT 模型转换。

```bash
python tools/deploy.py \
    configs/mmseg/segmentation_tensorrt_static-512x512.py \
    ../atl_config.py \
    ../deeplabv3plus_r18-d8_512x512_80k_potsdam_20211219_020601-75fd5bc3.pth \
    ../2_13_3584_2560_4096_3072.png \
    --work-dir ../atl_trt_models \
    --device cuda:0 \
    --show \
    --dump-info

```

转换成功后您将看到以下信息及 TensorRT 模型文件夹：

```bash
10/09 20:15:50 - mmengine - INFO - visualize pytorch model success.
10/09 20:15:50 - mmengine - INFO - All process success.
```

<div align="center">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/2ac1428f-b787-4fdd-beaf-6397e5b21e33" alt="NVIDIA-Jetson" width="340">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/70470a39-6a4f-4fd5-a06d-9b9d59a768ef" alt="NVIDIA-Jetson" width="200">
</div>

## 6.3 模型测速

执行以下命令完成模型测速，详细内容请查看[ profiler ](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/02-how-to-run/useful_tools.md#profiler)

```bash
python tools/profiler.py \
    ${DEPLOY_CFG} \
    ${MODEL_CFG} \
    ${IMAGE_DIR} \
    --model ${MODEL} \
    --device ${DEVICE} \
    --shape ${SHAPE} \
    --num-iter ${NUM_ITER} \
    --warmup ${WARMUP} \
    --cfg-options ${CFG_OPTIONS} \
    --batch-size ${BATCH_SIZE} \
    --img-ext ${IMG_EXT}
```

示例：

```bash
python tools/profiler.py \
    configs/mmseg/segmentation_tensorrt_static-512x512.py \
    ../atl_config.py \
    ../atl_demo_img \
    --model /home/sirs/AI-Tianlong/OpenMMLab/atl_trt_models/end2end.engine \
    --device cuda:0 \
    --shape 512x512 \
    --num-iter 100
```

测速结果

![image](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/874e9742-ee10-490c-9e69-17da0096c49b)

## 6.4 模型推理

根据[6.2.2](#6.2.2-TensorRT-模型转换)中生成的TensorRT模型文件夹，进行模型推理。

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg='./mmdeploy/configs/mmseg/segmentation_tensorrt_static-512x512.py'
model_cfg='./atl_config.py'
device='cuda:0'
backend_model = ['./atl_trt_models/end2end.engine']
image = './atl_demo_img/2_13_2048_1024_2560_1536.png'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='./output_segmentation.png')
```

即可得到推理结果：

<div align="center">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/d0ae1fa8-e223-4b3f-b699-6bfa8db38133" alt="NVIDIA-Jetson" width="40%">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/6d999cbe-2101-4e1b-b4a9-13115c9d1928" alt="NVIDIA-Jetson" width="40%">
</div>
