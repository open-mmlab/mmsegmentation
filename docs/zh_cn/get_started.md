# 依赖

在本节中，我们将演示如何用PyTorch准备一个环境。

MMSegmentation 可以在 Linux、Windows 和 MacOS 上运行。它需要 Python 3.6 以上，CUDA 9.2 以上和 PyTorch 1.3 以上。

```{note}
如果您对PyTorch有经验并且已经安装了它，请跳到下一节。否则，您可以按照以下步骤进行准备。
```

**第一步** 从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda。

**第二步** 创建并激活一个 conda 环境。

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**第三步** 按照[官方说明](https://pytorch.org/get-started/locally/)安装 PyTorch。

在 GPU 平台上：

```shell
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上：

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# 安装

我们建议用户遵循我们的最佳实践来安装MMSegmentation，同时整个过程是高度可定制的。更多信息见[自定义安装](#customize-installation)部分。

## 最佳实践

**第一步** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv)

```shell
pip install -U openmim
mim install mmcv-full
```

**第二步** 安装 MMSegmentation

根据具体需求，我们支持两种安装模式：

- [从源码安装（推荐）](#%E4%BB%8E%E6%BA%90%E7%A0%81%E5%AE%89%E8%A3%85)：如果基于 MMSegmentation 框架开发自己的任务，需要添加新的功能，比如新的模型或是数据集，或者使用我们提供的各种工具。
- [作为 Python 包安装](#%E4%BD%9C%E4%B8%BA-python-%E5%8C%85%E5%AE%89%E8%A3%85)：只是希望调用 MMSegmentation 的接口，或者在自己的项目中导入 MMSegmentation 中的模块。

### 从源码安装

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
# "-v "指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

### 作为 Python 包安装

```shell
pip install mmsegmentation
```

## 验证安装

为了验证 MMSegmentation 是否安装正确，我们提供了一些示例代码来执行模型推理。

**第一步** 我们需要下载配置文件和模型权重文件。

```shell
mim download mmsegmentation --config pspnet_r50-d8_512x1024_40k_cityscapes --dest .
```

下载将需要几秒钟或更长时间，这取决于你的网络环境。完成后，你会在当前文件夹中发现两个文件`pspnet_r50-d8_512x1024_40k_cityscapes.py`和`pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth`。

**第二步** 验证推理示例

如果您是**从源码安装**的 MMSegmentation，那么直接运行以下命令进行验证：

```shell
python demo/image_demo.py demo/demo.png pspnet_r50-d8_512x1024_40k_cityscapes.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cpu --out-file result.jpg
```

你会在你的当前文件夹中看到一个新的图像`result.jpg`，其中的分割掩膜覆盖在所有对象上。

如果您是**作为 PyThon 包安装**，那么可以打开您的 Python 解释器，复制并粘贴如下代码：

```python
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# 通过配置文件和模型权重文件构建模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# 对单张图片进行推理并展示结果
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)
# 在新窗口中可视化推理结果
model.show_result(img, result, show=True)
# 或将可视化结果存储在文件中
# 你可以修改 opacity 在(0,1]之间的取值来改变绘制好的分割图的透明度
model.show_result(img, result, out_file='result.jpg', opacity=0.5)

# 对视频进行推理并展示结果
video = mmcv.VideoReader('video.mp4')
for frame in video:
   result = inference_segmentor(model, frame)
   model.show_result(frame, result, wait_time=1)
```

你可以修改上面的代码来测试一张图片或一段视频，这两种方式都可以验证安装是否成功。

## 自定义安装

### CUDA 版本

在安装 PyTorch 时，您需要指定 CUDA 的版本。如果您不清楚应该选择哪一个，请遵循我们的建议。

- 对于 Ampere 架构的 NVIDIA GPU，例如 GeForce 30 系列 以及 NVIDIA A100，CUDA 11 是必需的。
- 对于更早的 NVIDIA GPU，CUDA 11 是向后兼容 (backward compatible) 的，但 CUDA 10.2 能够提供更好的兼容性，也更加轻量。

请确保您的 GPU 驱动版本满足最低的版本需求，参阅[这张表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

```{note}
如果按照我们的最佳实践进行安装，CUDA 运行时库就足够了，因为我们提供相关 CUDA 代码的预编译，您不需要进行本地编译。
但如果您希望从源码进行 MMCV 的编译，或是进行其他 CUDA 算子的开发，那么就必须安装完整的 CUDA 工具链，参见
[NVIDIA 官网](https://developer.nvidia.com/cuda-downloads)，另外还需要确保该 CUDA 工具链的版本与 PyTorch 安装时
的配置相匹配（如用 `conda install` 安装 PyTorch 时指定的 cudatoolkit 版本）。
```

### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此其对 PyTorch 的依赖比较复杂。MIM 会自动解析这些
依赖，选择合适的 MMCV 预编译包，使安装更简单，但它并不是必需的。

要使用 pip 而不是 MIM 来安装 MMCV，请遵照 [MMCV 安装指南](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html)。
它需要您用指定 url 的形式手动指定对应的 PyTorch 和 CUDA 版本。

举个例子，如下命令将会安装基于 PyTorch 1.10.x 和 CUDA 11.3 编译的 mmcv-full。

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### 在 CPU 环境中安装

MMPose 可以仅在 CPU 环境中安装，在 CPU 模式下，您可以完成训练（需要 MMCV 版本 >= 1.4.4）、测试和模型推理等所有操作。

### 在 Google Colab 中安装

[Google Colab](https://colab.research.google.com/) 通常已经包含了 PyTorch 环境，因此我们只需要安装 MMCV 和 MMPose 即可，命令如下：

**第一步** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv)

```shell
!pip3 install openmim
!mim install mmcv-full
```

**第二步** 从源码安装 MMSegmentation

```shell
!git clone https://github.com/open-mmlab/mmsegmentation.git
%cd mmsegmentation
!pip install -e .
```

**第三步** 验证

```python
import mmseg
print(mmseg.__version__)
# 预期输出：0.24.1 或其他版本号
```

```{note}
在 Jupyter 中，感叹号 `!` 用于执行外部命令，而 `%cd` 是一个[魔术命令](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd)，用于切换 Python 的工作路径。
```

### 通过 Docker 使用 MMSegmentation

我们提供了一个[Dockerfile](https://github.com/open-mmlab/mmsegmentation/blob/master/docker/Dockerfile)来构建一个镜像。请确保你的[docker版本](https://docs.docker.com/engine/install/) >=19.03。

```shell
# build an image with PyTorch 1.11, CUDA 11.3
# If you prefer other versions, just modified the Dockerfile
docker build -t mmsegmentation docker/
```

用以下命令运行 Docker 镜像：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmpose/data mmpose
```

## 故障解决

如果你在安装过程中遇到一些问题，请先查看[FAQ](faq.md)页面。

如果没有找到解决方案，你也可以在GitHub上[打开一个问题](https://github.com/open-mmlab/mmsegmentation/issues/new/choose)。
