# 教程5：模型部署

# MMSegmentation 模型部署

- [教程5：模型部署](#教程5模型部署)
- [MMSegmentation 模型部署](#mmsegmentation-模型部署)
  - [安装](#安装)
    - [安装 mmseg](#安装-mmseg)
    - [安装 mmdeploy](#安装-mmdeploy)
  - [模型转换](#模型转换)
  - [模型规范](#模型规范)
  - [模型推理](#模型推理)
    - [后端模型推理](#后端模型推理)
    - [SDK 模型推理](#sdk-模型推理)
  - [模型支持列表](#模型支持列表)
  - [注意事项](#注意事项)

______________________________________________________________________

[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main) 又称`mmseg`，是一个基于 PyTorch 的开源对象分割工具箱。它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

## 安装

### 安装 mmseg

请参考[官网安装指南](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)。

### 安装 mmdeploy

mmdeploy 有以下几种安装方式:

**方式一：** 安装预编译包

请参考[安装概述](https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html#mmdeploy)

**方式二：** 一键式脚本安装

如果部署平台是 **Ubuntu 18.04 及以上版本**， 请参考[脚本安装说明](../01-how-to-build/build_from_script.md)，完成安装过程。
比如，以下命令可以安装 mmdeploy 以及配套的推理引擎——`ONNX Runtime`.

```shell
git clone --recursive -b main https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

**说明**:

- 把 `$(pwd)/build/lib` 添加到 `PYTHONPATH`，目的是为了加载 mmdeploy SDK python 包 `mmdeploy_runtime`，在章节 [SDK模型推理](#sdk模型推理)中讲述其用法。
- 在[使用 ONNX Runtime推理后端模型](#后端模型推理)时，需要加载自定义算子库，需要把 ONNX Runtime 库的路径加入环境变量 `LD_LIBRARY_PATH`中。
  **方式三：** 源码安装

在方式一、二都满足不了的情况下，请参考[源码安装说明](../01-how-to-build/build_from_source.md) 安装 mmdeploy 以及所需推理引擎。

## 模型转换

你可以使用 [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/tree/main/tools/deploy.py) 把 mmseg 模型一键式转换为推理后端模型。
该工具的详细使用说明请参考[这里](https://github.com/open-mmlab/mmdeploy/tree/main/docs/en/02-how-to-run/convert_model.md#usage).

以下，我们将演示如何把 `unet` 转换为 onnx 模型。

```shell
cd mmdeploy

# download unet model from mmseg model zoo
mim download mmsegmentation --config unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024 --dest .

# convert mmseg model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmseg/segmentation_onnxruntime_dynamic.py \
    unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py \
    fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth \
    demo/resources/cityscapes.png \
    --work-dir mmdeploy_models/mmseg/ort \
    --device cpu \
    --show \
    --dump-info
```

转换的关键之一是使用正确的配置文件。项目中已内置了各后端部署[配置文件](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmseg)。
文件的命名模式是：

```
segmentation_{backend}-{precision}_{static | dynamic}_{shape}.py
```

其中：

- **{backend}:** 推理后端名称。比如，onnxruntime、tensorrt、pplnn、ncnn、openvino、coreml 等等
- **{precision}:** 推理精度。比如，fp16、int8。不填表示 fp32
- **{static | dynamic}:** 动态、静态 shape
- **{shape}:** 模型输入的 shape 或者 shape 范围

在上例中，你也可以把 `unet` 转为其他后端模型。比如使用`segmentation_tensorrt-fp16_dynamic-512x1024-2048x2048.py`，把模型转为 tensorrt-fp16 模型。

```{tip}
当转 tensorrt 模型时, --device 需要被设置为 "cuda"
```

## 模型规范

在使用转换后的模型进行推理之前，有必要了解转换结果的结构。 它存放在 `--work-dir` 指定的路路径下。

上例中的`mmdeploy_models/mmseg/ort`，结构如下：

```
mmdeploy_models/mmseg/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

重要的是：

- **end2end.onnx**: 推理引擎文件。可用 ONNX Runtime 推理
- \***.json**:  mmdeploy SDK 推理所需的 meta 信息

整个文件夹被定义为**mmdeploy SDK model**。换言之，**mmdeploy SDK model**既包括推理引擎，也包括推理 meta 信息。

## 模型推理

### 后端模型推理

以上述模型转换后的 `end2end.onnx` 为例，你可以使用如下代码进行推理：

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmseg/segmentation_onnxruntime_dynamic.py'
model_cfg = './unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmseg/ort/end2end.onnx']
image = './demo/resources/cityscapes.png'

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

### SDK 模型推理

你也可以参考如下代码，对 SDK model 进行推理：

```python
from mmdeploy_runtime import Segmentor
import cv2
import numpy as np

img = cv2.imread('./demo/resources/cityscapes.png')

# create a classifier
segmentor = Segmentor(model_path='./mmdeploy_models/mmseg/ort', device_name='cpu', device_id=0)
# perform inference
seg = segmentor(img)

# visualize inference result
## random a palette with size 256x3
palette = np.random.randint(0, 256, size=(256, 3))
color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
for label, color in enumerate(palette):
    color_seg[seg == label, :] = color
# convert to BGR
color_seg = color_seg[..., ::-1]
img = img * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)
cv2.imwrite('output_segmentation.png', img)
```

除了python API，mmdeploy SDK 还提供了诸如 C、C++、C#、Java等多语言接口。
你可以参考[样例](https://github.com/open-mmlab/mmdeploy/tree/main/demo)学习其他语言接口的使用方法。

## 模型支持列表

| Model                                                                                                     | TorchScript | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVino |
| :-------------------------------------------------------------------------------------------------------- | :---------: | :---------: | :------: | :--: | :---: | :------: |
| [FCN](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/fcn)                                 |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [PSPNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/pspnet)[\*](#static_shape)        |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [DeepLabV3](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/deeplabv3)                     |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [DeepLabV3+](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/deeplabv3plus)                |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [Fast-SCNN](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/fastscnn)[\*](#static_shape)   |      Y      |      Y      |    Y     |  N   |   Y   |    Y     |
| [UNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet)                               |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [ANN](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/ann)[\*](#static_shape)              |      Y      |      Y      |    Y     |  N   |   N   |    N     |
| [APCNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/apcnet)                           |      Y      |      Y      |    Y     |  Y   |   N   |    N     |
| [BiSeNetV1](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/bisenetv1)                     |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [BiSeNetV2](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/bisenetv2)                     |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [CGNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/cgnet)                             |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [DMNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/dmnet)                             |      ?      |      Y      |    N     |  N   |   N   |    N     |
| [DNLNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/dnlnet)                           |      ?      |      Y      |    Y     |  Y   |   N   |    Y     |
| [EMANet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/emanet)                           |      Y      |      Y      |    Y     |  N   |   N   |    Y     |
| [EncNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/encnet)                           |      Y      |      Y      |    Y     |  N   |   N   |    Y     |
| [ERFNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/erfnet)                           |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [FastFCN](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/fastfcn)                         |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [GCNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/gcnet)                             |      Y      |      Y      |    Y     |  N   |   N   |    N     |
| [ICNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/icnet)[\*](#static_shape)          |      Y      |      Y      |    Y     |  N   |   N   |    Y     |
| [ISANet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/isanet)[\*](#static_shape)        |      N      |      Y      |    Y     |  N   |   N   |    Y     |
| [NonLocal Net](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/nonlocal_net)               |      ?      |      Y      |    Y     |  Y   |   N   |    Y     |
| [OCRNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/ocrnet)                           |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [PointRend](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/point_rend)[\*](#static_shape) |      Y      |      Y      |    Y     |  N   |   N   |    N     |
| [Semantic FPN](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/sem_fpn)                    |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [STDC](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/stdc)                               |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [UPerNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/upernet)[\*](#static_shape)      |      N      |      Y      |    Y     |  N   |   N   |    N     |
| [DANet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/danet)                             |      ?      |      Y      |    Y     |  N   |   N   |    Y     |
| [Segmenter](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segmenter)[\*](#static_shape)  |      N      |      Y      |    Y     |  Y   |   N   |    Y     |
| [SegFormer](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer)[\*](#static_shape)  |      ?      |      Y      |    Y     |  N   |   N   |    Y     |
| [SETR](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/setr)                               |      ?      |      Y      |    N     |  N   |   N   |    Y     |
| [CCNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/ccnet)                             |      ?      |      N      |    N     |  N   |   N   |    N     |
| [PSANet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/psanet)                           |      ?      |      N      |    N     |  N   |   N   |    N     |
| [DPT](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/dpt)                                 |      ?      |      N      |    N     |  N   |   N   |    N     |

## 注意事项

- 所有 mmseg 模型仅支持 "whole" 推理模式。

- <i id=“static_shape”>PSPNet，Fast-SCNN</i> 仅支持静态输入，因为多数推理框架的 [nn.AdaptiveAvgPool2d](https://github.com/open-mmlab/mmsegmentation/blob/0c87f7a0c9099844eff8e90fa3db5b0d0ca02fee/mmseg/models/decode_heads/psp_head.py#L38) 不支持动态输入。

- 对于仅支持静态形状的模型，应使用静态形状的部署配置文件，例如 `configs/mmseg/segmentation_tensorrt_static-1024x2048.py`

- 对于喜欢部署模型生成概率特征图的用户，将 `codebase_config = dict(with_argmax=False)` 放在部署配置中就足够了。
