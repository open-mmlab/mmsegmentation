# 开始：安装和运行 MMSeg

## 预备知识

本教程中，我们将会演示如何使用 PyTorch 准备环境。

MMSegmentation 可以在 Linux, Windows 和 macOS 系统上运行，并且需要安装 Python 3.6+, CUDA 9.2+ 和 PyTorch 1.5+

**注意:**
如果您已经安装了 PyTorch, 可以跳过该部分，直接到[下一小节](##安装)。否则，您可以按照以下步骤操作。

**步骤 0.** 从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda

**步骤 1.** 创建一个 conda 环境，并激活

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** 参考 [official instructions](https://pytorch.org/get-started/locally/) 安装 PyTorch

在 GPU 平台上：

```shell
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## 安装

我们建议用户遵循我们的最佳实践来安装 MMSegmentation 。但是整个过程是高度自定义的。更多信息请参见[自定义安装](##自定义安装)部分。

### 最佳实践

**步骤 0.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv)

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**步骤 1.** 安装 MMSegmentation

情况 a: 如果您想立刻开发和运行 mmsegmentation，您可通过源码安装：

```shell
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
# '-v' 表示详细模式，更多的输出
# '-e' 表示以可编辑模式安装工程，
# 因此对代码所做的任何修改都生效，无需重新安装
```

情况 b: 如果您把 mmsegmentation 作为依赖库或者第三方库，可以通过 pip 安装：

```shell
pip install "mmsegmentation>=1.0.0"
```

### 验证是否安装成功

为了验证 MMSegmentation 是否正确安装，我们提供了一些示例代码来运行一个推理 demo 。

**步骤 1.** 下载配置文件和模型文件

```shell
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
```

该下载过程可能需要花费几分钟，这取决于您的网络环境。当下载结束，您将看到以下两个文件在您当前工作目录：`pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py` 和 `pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth`

**步骤 2.** 验证推理 demo

选项 (a). 如果您通过源码安装了 mmsegmentation，运行以下命令即可：

```shell
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```

您将在当前文件夹中看到一个新图像 `result.jpg`，其中所有目标都覆盖了分割 mask

选项 (b). 如果您通过 pip 安装 mmsegmentation, 打开您的 python 解释器，复制粘贴以下代码：

```python
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# 根据配置文件和模型文件建立模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 在单张图像上测试并可视化
img = 'demo/demo.png'  # or img = mmcv.imread(img), 这样仅需下载一次
result = inference_model(model, img)
# 在新的窗口可视化结果
show_result_pyplot(model, img, result, show=True)
# 或者将可视化结果保存到图像文件夹中
# 您可以修改分割 map 的透明度 (0, 1].
show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)
# 在一段视频上测试并可视化分割结果
video = mmcv.VideoReader('video.mp4')
for frame in video:
   result = inference_segmentor(model, frame)
   show_result_pyplot(model, result, wait_time=1)
```

您可以修改上面的代码来测试单个图像或视频，这两个选项都可以验证安装是否成功。

### 自定义安装

#### CUDA 版本

当安装 PyTorch 的时候，您需要指定 CUDA 的版本， 如果您不确定选择哪个版本，请遵循我们的建议：

- 对于基于 Ampere 的 NVIDIA GPUs, 例如 GeForce 30 系列和 NVIDIA A100, 必须要求是 CUDA 11.
- 对于更老的 NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 提供了更好的兼容性，以及更加的轻量化

请确保 GPU 驱动满足最小的版本需求。详情请参考这个[表格](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)

**注意:**
如果您按照我们的最佳实践，安装 CUDA 运行库就足够了，因为不需要 CUDA 代码在本地编译。 但是如果您希望从源码编译 MMCV 或者需要开发其他的 CUDA 算子，您需要从 NVIDIA 的[官网](https://developer.nvidia.com/cuda-downloads)安装完整的 CUDA 工具，同时它的版本需要与 PyTorch 的 CUDA 版本匹配。即 `conda install` 命令中指定的 cudatoolkit 版本。

#### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此与 PyTorch 的依赖方式比较复杂。MIM 自动解决了这种依赖关系，使安装更容易。然而，MIM 也并不是必须的。

为了使用 pip 而不是 MIM 安装 MMCV, 请参考 [MMCV 安装指南](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). 这需要手动指定一个基于 PyTorch 版本及其 CUDA 版本的 find-url.

例如，以下命令可为 PyTorch 1.10.x and CUDA 11.3 安装 mmcv==2.0.0

```shell
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

#### 在仅有 CPU 的平台安装

MMSegmentation 可以在仅有 CPU 的版本上运行。在 CPU 模式，您可以训练（需要 MMCV 版本 >= 2.0.0），测试和推理模型。

#### 在 Google Colab 上安装

[Google Colab](https://research.google.com/) 通常已经安装了 PyTorch，因此我们仅需要通过以下命令安装 MMCV 和 MMSegmentation。

**步骤 1.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv)

```shell
!pip3 install openmim
!mim install mmengine
!mim install "mmcv>=2.0.0"
```

**Step 2.** 通过源码安装 MMSegmentation

```shell
!git clone https://github.com/open-mmlab/mmsegmentation.git
%cd mmsegmentation
!git checkout main
!pip install -e .
```

**Step 3.** 验证

```python
import mmseg
print(mmseg.__version__)
# 示例输出: 1.0.0
```

**注意:**
在 Jupyter 中, 感叹号 `!` 用于调用外部可执行命令，`%cd` 是一个 [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) 可以改变当前 python 的工作目录。

### 通过 Docker 使用 MMSegmentation

我们提供了一个 [Dockerfile](https://github.com/open-mmlab/mmsegmentation/blob/master/docker/Dockerfile) 来建立映像。确保您的 [docker 版本](https://docs.docker.com/engine/install/) >=19.03.

```shell
# 通过 PyTorch 1.11, CUDA 11.3 建立映像
# 如果您使用其他版本，修改 Dockerfile 即可
docker build -t mmsegmentation docker/
```

运行：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmsegmentation/data mmsegmentation
```

### 可选依赖

#### 安装 GDAL

[GDAL](https://gdal.org/) 是一个用于栅格和矢量地理空间数据格式的转换库。安装 GDAL 可以读取复杂格式和极大的遥感图像。

```shell
conda install GDAL
```

## 问题解答

如果您在安装过程中遇到了其他问题，请第一时间查阅 [FAQ](notes/faq.md) 文件。如果没有找到答案，您也可以在 GitHub 上提出 [issue](https://github.com/open-mmlab/mmsegmentation/issues/new/choose)
