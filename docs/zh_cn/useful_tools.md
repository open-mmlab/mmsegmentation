## 常用工具

除了训练和测试的脚本，我们在 `tools/` 文件夹路径下还提供许多有用的工具。

### 计算参数量（params）和计算量（ FLOPs） (试验性)

我们基于 [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch)
提供了一个用于计算给定模型参数量和计算量的脚本。

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

您将得到如下的结果：

```none
==============================
Input shape: (3, 2048, 1024)
Flops: 1429.68 GMac
Params: 48.98 M
==============================
```

**注意**: 这个工具仍然是试验性的，我们无法保证数字是正确的。您可以拿这些结果做简单的实验的对照，在写技术文档报告或者论文前您需要再次确认一下。

(1) 计算量与输入的形状有关，而参数量与输入的形状无关，默认的输入形状是 (1, 3, 1280, 800)；
(2) 一些运算操作，如 GN 和其他定制的运算操作没有加入到计算量的计算中。

### 发布模型

在您上传一个模型到云服务器之前，您需要做以下几步：
(1) 将模型权重转成 CPU 张量；
(2) 删除记录优化器状态 (optimizer states)的相关信息；
(3) 计算检查点文件 (checkpoint file) 的哈希编码（hash id）并且将哈希编码加到文件名中。

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

例如，

```shell
python tools/publish_model.py work_dirs/pspnet/latest.pth psp_r50_hszhao_200ep.pth
```

最终输出文件将是 `psp_r50_512x1024_40ki_cityscapes-{hash id}.pth`。

### 导出 ONNX (试验性)

我们提供了一个脚本来导出模型到 [ONNX](https://github.com/onnx/onnx) 格式。被转换的模型可以通过工具 [Netron](https://github.com/lutzroeder/netron)
来可视化。除此以外，我们同样支持对 PyTorch 和 ONNX 模型的输出结果做对比。

```bash
python tools/pytorch2onnx.py \
    ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --output-file ${ONNX_FILE} \
    --input-img ${INPUT_IMG} \
    --shape ${INPUT_SHAPE} \
    --rescale-shape ${RESCALE_SHAPE} \
    --show \
    --verify \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.mode="whole"
```

各个参数的描述:

- `config` : 模型配置文件的路径
- `--checkpoint` : 模型检查点文件的路径
- `--output-file`: 输出的 ONNX 模型的路径。如果没有专门指定，它默认是 `tmp.onnx`
- `--input-img` : 用来转换和可视化的一张输入图像的路径
- `--shape`: 模型的输入张量的高和宽。如果没有专门指定，它将被设置成 `test_pipeline` 的 `img_scale`
- `--rescale-shape`: 改变输出的形状。设置这个值来避免 OOM，它仅在 `slide` 模式下可以用
- `--show`: 是否打印输出模型的结构。如果没有被专门指定，它将被设置成 `False`
- `--verify`: 是否验证一个输出模型的正确性 (correctness)。如果没有被专门指定，它将被设置成 `False`
- `--dynamic-export`: 是否导出形状变化的输入与输出的 ONNX 模型。如果没有被专门指定，它将被设置成 `False`
- `--cfg-options`: 更新配置选项

**注意**: 这个工具仍然是试验性的，目前一些自定义操作还没有被支持

### 评估 ONNX 模型

我们提供 `tools/deploy_test.py` 去评估不同后端的 ONNX 模型。

#### 先决条件

- 安装 onnx 和 onnxruntime-gpu

  ```shell
  pip install onnx onnxruntime-gpu
  ```

- 参考 [如何在 MMCV 里构建 tensorrt 插件](https://mmcv.readthedocs.io/en/latest/tensorrt_plugin.html#how-to-build-tensorrt-plugins-in-mmcv) 安装TensorRT (可选)

#### 使用方法

```bash
python tools/deploy_test.py \
    ${CONFIG_FILE} \
    ${MODEL_FILE} \
    ${BACKEND} \
    --out ${OUTPUT_FILE} \
    --eval ${EVALUATION_METRICS} \
    --show \
    --show-dir ${SHOW_DIRECTORY} \
    --cfg-options ${CFG_OPTIONS} \
    --eval-options ${EVALUATION_OPTIONS} \
    --opacity ${OPACITY} \
```

各个参数的描述:

- `config`: 模型配置文件的路径
- `model`: 被转换的模型文件的路径
- `backend`: 推理的后端，可选项：`onnxruntime`， `tensorrt`
- `--out`: 输出结果成 pickle 格式文件的路径
- `--format-only` : 不评估直接给输出结果的格式。通常用在当您想把结果输出成一些测试服务器需要的特定格式时。如果没有被专门指定，它将被设置成 `False`。 注意这个参数是用 `--eval` 来 **手动添加**
- `--eval`: 评估指标，取决于每个数据集的要求，例如 "mIoU" 是大多数据集的指标而 "cityscapes" 仅针对 Cityscapes 数据集。注意这个参数是用 `--format-only` 来 **手动添加**
- `--show`: 是否展示结果
- `--show-dir`: 涂上结果的图像被保存的文件夹的路径
- `--cfg-options`: 重写配置文件里的一些设置，`xxx=yyy` 格式的键值对将被覆盖到配置文件里
- `--eval-options`: 自定义的评估的选项， `xxx=yyy` 格式的键值对将成为  `dataset.evaluate()` 函数的参数变量
- `--opacity`: 涂上结果的分割图的透明度，范围在 (0, 1\] 之间

#### 结果和模型

|    模型    |                    配置文件                     |   数据集   | 评价指标 | PyTorch | ONNXRuntime | TensorRT-fp32 | TensorRT-fp16 |
| :--------: | :---------------------------------------------: | :--------: | :------: | :-----: | :---------: | :-----------: | :-----------: |
|    FCN     |      fcn_r50-d8_512x1024_40k_cityscapes.py      | cityscapes |   mIoU   |  72.2   |    72.2     |     72.2      |     72.2      |
|   PSPNet   |    pspnet_r50-d8_512x1024_40k_cityscapes.py     | cityscapes |   mIoU   |  77.8   |    77.8     |     77.8      |     77.8      |
| deeplabv3  |   deeplabv3_r50-d8_512x1024_40k_cityscapes.py   | cityscapes |   mIoU   |  79.0   |    79.0     |     79.0      |     79.0      |
| deeplabv3+ | deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py | cityscapes |   mIoU   |  79.6   |    79.5     |     79.5      |     79.5      |
|   PSPNet   |     pspnet_r50-d8_769x769_40k_cityscapes.py     | cityscapes |   mIoU   |  78.2   |    78.1     |               |               |
| deeplabv3  |   deeplabv3_r50-d8_769x769_40k_cityscapes.py    | cityscapes |   mIoU   |  78.5   |    78.3     |               |               |
| deeplabv3+ | deeplabv3plus_r50-d8_769x769_40k_cityscapes.py  | cityscapes |   mIoU   |  78.9   |    78.7     |               |               |

**注意**: TensorRT 仅在使用 `whole mode` 测试模式时的配置文件里可用。

### 导出 TorchScript (试验性)

我们同样提供一个脚本去把模型导出成 [TorchScript](https://pytorch.org/docs/stable/jit.html) 格式。您可以使用 pytorch C++ API [LibTorch](https://pytorch.org/docs/stable/cpp_index.html) 去推理训练好的模型。
被转换的模型能被像 [Netron](https://github.com/lutzroeder/netron) 的工具来可视化。此外，我们还支持 PyTorch 和 TorchScript 模型的输出结果的比较。

```shell
python tools/pytorch2torchscript.py \
    ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --output-file ${ONNX_FILE}
    --shape ${INPUT_SHAPE}
    --verify \
    --show
```

各个参数的描述:

- `config` : pytorch 模型的配置文件的路径
- `--checkpoint` : pytorch 模型的检查点文件的路径
- `--output-file`: TorchScript 模型输出的路径，如果没有被专门指定，它将被设置成 `tmp.pt`
- `--input-img` : 用来转换和可视化的输入图像的路径
- `--shape`: 模型的输入张量的宽和高。如果没有被专门指定，它将被设置成 `512 512`
- `--show`: 是否打印输出模型的追踪图 (traced graph)，如果没有被专门指定，它将被设置成 `False`
- `--verify`: 是否验证一个输出模型的正确性 (correctness)，如果没有被专门指定，它将被设置成 `False`

**注意**: 目前仅支持 PyTorch>=1.8.0 版本

**注意**: 这个工具仍然是试验性的，一些自定义操作符目前还不被支持

例子:

- 导出 PSPNet 在 cityscapes 数据集上的 pytorch 模型

  ```shell
  python tools/pytorch2torchscript.py configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
  --checkpoint checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
  --output-file checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pt \
  --shape 512 1024
  ```

### 导出 TensorRT (试验性)

一个导出 [ONNX](https://github.com/onnx/onnx) 模型成 [TensorRT](https://developer.nvidia.com/tensorrt) 格式的脚本

先决条件

- 按照 [ONNXRuntime in mmcv](https://mmcv.readthedocs.io/en/latest/deployment/onnxruntime_op.html) 和 [TensorRT plugin in mmcv](https://github.com/open-mmlab/mmcv/blob/master/docs/en/deployment/tensorrt_plugin.md) ，用 ONNXRuntime 自定义运算 (custom ops) 和 TensorRT 插件安装 `mmcv-full`
- 使用 [pytorch2onnx](#convert-to-onnx-experimental) 将模型从 PyTorch 转成 ONNX

使用方法

```bash
python ${MMSEG_PATH}/tools/onnx2tensorrt.py \
    ${CFG_PATH} \
    ${ONNX_PATH} \
    --trt-file ${OUTPUT_TRT_PATH} \
    --min-shape ${MIN_SHAPE} \
    --max-shape ${MAX_SHAPE} \
    --input-img ${INPUT_IMG} \
    --show \
    --verify
```

各个参数的描述:

- `config` : 模型的配置文件
- `model` : 输入的 ONNX 模型的路径
- `--trt-file` : 输出的 TensorRT 引擎的路径
- `--max-shape` : 模型的输入的最大形状
- `--min-shape` : 模型的输入的最小形状
- `--fp16` : 做 fp16 模型转换
- `--workspace-size` : 在 GiB 里的最大工作空间大小 (Max workspace size)
- `--input-img` : 用来可视化的图像
- `--show` : 做结果的可视化
- `--dataset` : Palette provider, 默认为 `CityscapesDataset`
- `--verify` : 验证 ONNXRuntime 和 TensorRT 的输出
- `--verbose` : 当创建 TensorRT 引擎时，是否详细做信息日志。默认为 False

**注意**: 仅在全图测试模式 (whole mode) 下测试过

## 其他内容

### 打印完整的配置文件

`tools/print_config.py` 会逐字逐句的打印整个配置文件，展开所有的导入。

```shell
python tools/print_config.py \
  ${CONFIG} \
  --graph \
  --cfg-options ${OPTIONS [OPTIONS...]} \
```

各个参数的描述:

- `config` : pytorch 模型的配置文件的路径
- `--graph` : 是否打印模型的图 (models graph)
- `--cfg-options`: 自定义替换配置文件的选项

### 对训练日志 (training logs) 画图

`tools/analyze_logs.py` 会画出给定的训练日志文件的 loss/mIoU 曲线，首先需要 `pip install seaborn` 安装依赖包。

```shell
python tools/analyze_logs.py xxx.log.json [--keys ${KEYS}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

示例:

- 对 mIoU, mAcc, aAcc 指标画图

  ```shell
  python tools/analyze_logs.py log.json --keys mIoU mAcc aAcc --legend mIoU mAcc aAcc
  ```

- 对 loss 指标画图

  ```shell
  python tools/analyze_logs.py log.json --keys loss --legend loss
  ```

### 转换其他仓库的权重

`tools/model_converters/` 提供了若干个预训练权重转换脚本，支持将其他仓库的预训练权重的 key 转换为与 MMSegmentation 相匹配的 key。

#### ViT Swin MiT Transformer 模型

- ViT

`tools/model_converters/vit2mmseg.py` 将 timm 预训练模型转换到 MMSegmentation。

```shell
python tools/model_converters/vit2mmseg.py ${SRC} ${DST}
```

- Swin

  `tools/model_converters/swin2mmseg.py` 将官方预训练模型转换到 MMSegmentation。

  ```shell
  python tools/model_converters/swin2mmseg.py ${SRC} ${DST}
  ```

- SegFormer

  `tools/model_converters/mit2mmseg.py` 将官方预训练模型转换到 MMSegmentation。

  ```shell
  python tools/model_converters/mit2mmseg.py ${SRC} ${DST}
  ```

## 模型服务

为了用 [`TorchServe`](https://pytorch.org/serve/) 服务 `MMSegmentation` 的模型 ， 您可以遵循如下流程:

### 1. 将 model 从　MMSegmentation 转换到 TorchServe

```shell
python tools/mmseg2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output-folder ${MODEL_STORE} \
--model-name ${MODEL_NAME}
```

**注意**: ${MODEL_STORE} 需要设置为某个文件夹的绝对路径

### 2. 构建 `mmseg-serve` 容器镜像 (docker image)

```shell
docker build -t mmseg-serve:latest docker/serve/
```

### 3. 运行 `mmseg-serve`

请查阅官方文档: [使用容器运行 TorchServe](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment)

为了在 GPU 环境下使用, 您需要安装 [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). 若在 CPU 环境下使用，您可以忽略添加 `--gpus` 参数。

示例:

```shell
docker run --rm \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=$MODEL_STORE,target=/home/model-server/model-store \
mmseg-serve:latest
```

阅读关于推理 (8080), 管理 (8081) 和指标 (8082) APIs 的 [文档](https://github.com/pytorch/serve/blob/072f5d088cce9bb64b2a18af065886c9b01b317b/docs/rest_api.md) 。

### 4. 测试部署

```shell
curl -O https://raw.githubusercontent.com/open-mmlab/mmsegmentation/master/resources/3dogs.jpg
curl http://127.0.0.1:8080/predictions/${MODEL_NAME} -T 3dogs.jpg -o 3dogs_mask.png
```

得到的响应将是一个 ".png" 的分割掩码.

您可以按照如下方法可视化输出:

```python
import matplotlib.pyplot as plt
import mmcv
plt.imshow(mmcv.imread("3dogs_mask.png", "grayscale"))
plt.show()
```

看到的东西将会和下图类似:

![3dogs_mask](../../resources/3dogs_mask.png)

然后您可以使用 `test_torchserve.py` 比较 torchserve 和 pytorch 的结果，并将它们可视化。

```shell
python tools/torchserve/test_torchserve.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MODEL_NAME}
[--inference-addr ${INFERENCE_ADDR}] [--result-image ${RESULT_IMAGE}] [--device ${DEVICE}]
```

示例：

```shell
python tools/torchserve/test_torchserve.py \
demo/demo.png \
configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py \
checkpoint/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth \
fcn
```
