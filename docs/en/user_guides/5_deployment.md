# Tutorial 5: Model Deployment

# MMSegmentation Model Deployment

- [Tutorial 5: Model Deployment](#tutorial-5-model-deployment)
- [MMSegmentation Model Deployment](#mmsegmentation-model-deployment)
  - [Installation](#installation)
    - [Install mmseg](#install-mmseg)
    - [Install mmdeploy](#install-mmdeploy)
  - [Convert model](#convert-model)
  - [Model specification](#model-specification)
  - [Model inference](#model-inference)
    - [Backend model inference](#backend-model-inference)
    - [SDK model inference](#sdk-model-inference)
  - [Supported models](#supported-models)
  - [Note](#note)

______________________________________________________________________

[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main), also known as `mmseg`, is an open source semantic segmentation toolbox based on Pytorch. It's a part of the [OpenMMLab](<(https://openmmlab.com/)>) object.

## Installation

### Install mmseg

Please follow the [Installation Guide](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

### Install mmdeploy

`mmdeploy` can be installed as follows:

**Option 1:** Install precompiled package

Please follow the [Installation overview](https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html#mmdeploy)

**Option 2:**  Automatic Installation script

If the deployment platform is **Ubuntu 18.04 +**, please follow the [scription installation](../01-how-to-build/build_from_script.md) to install.
For example, the following commands describe how to install mmdeploy and inference engine-`ONNX Runtime`.

```shell
git clone --recursive -b main https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

**NOTE**:

- Add `$(pwd)/build/lib` to `PYTHONPATH`, can loading mmdeploy SDK python package `mmdeploy_runtime`. See [SDK model inference](#SDK-model-inference) for more information.
- With [ONNX Runtime model inference](#Backend-model-inference), need to load custom operator library and add ONNX Runtime Library's PATH to `LD_LIBRARY_PATH`.

**Option 3:**  Install with mim

1. Use mim to install mmcv

```shell
pip install -U openmim
mim install "mmcv>=2.0.0rc2"
```

2. Install mmdeploy

```shell
git clone https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
mim install -e .
```

**Option 4:**  Build MMDeploy from source

If the first three methods aren't suitable, please [Build MMDeploy from source](<(../01-how-to-build/build_from_source.md)>)

## Convert model

[tools/deploy.py](https://github.com/open-mmlab/mmdeploy/tree/main/tools/deploy.py) can convert mmseg Model to backend model conveniently. See [this](https://github.com/open-mmlab/mmdeploy/tree/main/docs/en/02-how-to-run/convert_model.md#usage) for detailed information.

Then convert `unet` to onnx model as follows:

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

It is crucial to specify the correct deployment config during model conversion. MMDeploy has already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmseg) of all supported backends for mmsegmentation, under which the config file path follows the pattern:

```
segmentation_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml etc.
- **{precision}:** fp16, int8. When it's empty, it means fp32
- **{static | dynamic}:** static shape or dynamic shape
- **{shape}:** input shape or shape range of a model

Therefore, in the above example, you can also convert `unet` to tensorrt-fp16 model by `segmentation_tensorrt-fp16_dynamic-512x1024-2048x2048.py`.

```{tip}
When converting mmsegmentation models to tensorrt models, --device should be set to "cuda"
```

## Model specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmseg/ort` in the previous example. It includes:

```
mmdeploy_models/mmseg/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- ***xxx*.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmseg/ort** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model inference

### Backend model inference

Take the previous converted `end2end.onnx` model as an example, you can use the following code to inference the model and visualize the results:

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

### SDK model inference

You can also perform SDK model inference like following:

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

Besides python API, mmdeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demo](https://github.com/open-mmlab/mmdeploy/tree/main/demo)

## Supported models

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

## Note

- All mmseg models only support the 'whole' inference mode.

- <i id=“static_shape”>PSPNet，Fast-SCNN</i> only supports static input, because most inference framework's [nn.AdaptiveAvgPool2d](https://github.com/open-mmlab/mmsegmentation/blob/0c87f7a0c9099844eff8e90fa3db5b0d0ca02fee/mmseg/models/decode_heads/psp_head.py#L38) don't support dynamic input。

- For models that only support static shapes, should use the static shape deployment config file, such as `configs/mmseg/segmentation_tensorrt_static-1024x2048.py`

- To deploy models to generate probabilistic feature maps, please add `codebase_config = dict(with_argmax=False)` to deployment config file.
