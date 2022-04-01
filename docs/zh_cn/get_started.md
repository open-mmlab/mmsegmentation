## 依赖

- Linux or macOS (Windows下支持需要 mmcv-full，但运行时可能会有一些问题。)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (如果您基于源文件编译 PyTorch, CUDA 9.0也可以使用)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

可编译的 MMSegmentation 和 MMCV 版本如下所示，请对照对应版本安装以避免安装问题。

| MMSegmentation 版本 |          MMCV 版本         | MMClassification 版本    |
|:-----------------:|:--------------------------:|:------------------------:|
|      master       |  mmcv-full>=1.4.4, <=1.5.0 | mmcls>=0.20.1, <=1.0.0   |
|      0.23.0       |  mmcv-full>=1.4.4, <=1.5.0 | mmcls>=0.20.1, <=1.0.0   |
|      0.22.0       |  mmcv-full>=1.4.4, <=1.5.0 | mmcls>=0.20.1, <=1.0.0   |
|      0.21.1       |  mmcv-full>=1.4.4, <=1.5.0 | Not required             |
|      0.20.2       | mmcv-full>=1.3.13, <=1.5.0 | Not required             |
|      0.19.0       | mmcv-full>=1.3.13, <1.3.17 | Not required             |
|      0.18.0       | mmcv-full>=1.3.13, <1.3.17 | Not required             |
|      0.17.0       | mmcv-full>=1.3.7, <1.3.17  | Not required             |
|      0.16.0       | mmcv-full>=1.3.7, <1.3.17  | Not required             |
|      0.15.0       | mmcv-full>=1.3.7, <1.3.17  | Not required             |
|      0.14.1       | mmcv-full>=1.3.7, <1.3.17  | Not required             |
|      0.14.0       |  mmcv-full>=1.3.1, <1.3.2  | Not required             |
|      0.13.0       |  mmcv-full>=1.3.1, <1.3.2  | Not required             |
|      0.12.0       |  mmcv-full>=1.1.4, <1.3.2  | Not required             |
|      0.11.0       |  mmcv-full>=1.1.4, <1.3.0  | Not required             |
|      0.10.0       |  mmcv-full>=1.1.4, <1.3.0  | Not required             |
|       0.9.0       |  mmcv-full>=1.1.4, <1.3.0  | Not required             |
|       0.8.0       |  mmcv-full>=1.1.4, <1.2.0  | Not required             |
|       0.7.0       |  mmcv-full>=1.1.2, <1.2.0  | Not required             |
|       0.6.0       |  mmcv-full>=1.1.2, <1.2.0  | Not required             |

注意: 如果您已经安装好 mmcv， 您首先需要运行 `pip uninstall mmcv`。
如果 mmcv 和 mmcv-full 同时被安装，会报错 `ModuleNotFoundError`。

## 安装

a. 创建一个 conda 虚拟环境并激活它

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

```

b. 按照[官方教程](https://pytorch.org/) 安装 PyTorch 和 totchvision，
这里我们使用 PyTorch1.6.0 和 CUDA10.1，
您也可以切换至其他版本

```shell
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
```

c. 按照 [官方教程](https://mmcv.readthedocs.io/en/latest/#installation)
安装 [MMCV](https://mmcv.readthedocs.io/en/latest/) ，
`mmcv` 或 `mmcv-full` 和 MMSegmentation 均兼容，但对于 CCNet 和 PSANet，`mmcv-full` 里的 CUDA 运算是必须的

**在 Linux 下安装 mmcv：**

为了安装 MMCV, 我们推荐使用下面的这种预编译好的 MMCV.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

请替换 url 里面的 ``{cu_version}`` 和 ``{torch_version}`` 为您想要使用的版本. mmcv-full 仅在
PyTorch 1.x.0 上面编译, 因为在 1.x.0 和 1.x.1 之间通常是兼容的. 如果您的 PyTorch 版本是 1.x.1,
您可以安装用 PyTorch 1.x.0 编译的 mmcv-full 而它通常是可以正常使用的.
例如, 用 ``CUDA 10.1`` and ``PyTorch 1.6.0`` 安装使用 ``mmcv-full``, 使用如下命令:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6/index.html
```

请查看 [这里](https://github.com/open-mmlab/mmcv#installation) 来找到适配不同 PyTorch 和 CUDA 版本的 MMCV.

您也可以采用下面的命令来从源码编译 MMCV (可选)

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
# OR pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
cd ..
```

**重点:** 如果您已经安装了 MMCV, 您需要先运行 `pip uninstall mmcv`. 因为如果 `mmcv` 和 `mmcv-full` 被同时安装, 将会报错 `ModuleNotFoundError`.


**在 Windows 下安装 mmcv (有风险)：**

对于 Windows， MMCV 的安装需要本地 C++ 编译工具， 例如 cl.exe。 请添加编译工具至 %PATH%。

如果您已经在电脑上安装好Windows SDK 和 Visual Studio，cl.exe 的一个典型路径看起来如下：

```shell
C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.26.28801\bin\Hostx86\x64
```

或者您需要从网上下载 cl 编译工具并安装至路径。

随后，从 github 克隆 mmcv 并通过 pip 安装：

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -e .
```

或直接:

```shell
pip install mmcv
```

当前，mmcv-full 并不完全在 windows 上支持。

d. 安装 MMSegmentation

```shell
pip install mmsegmentation # 安装最新版本
```

或者

```shell
pip install git+https://github.com/open-mmlab/mmsegmentation.git # 安装 master 分支
```

此外，如果您想安装 `dev` 模式的 MMSegmentation, 运行如下命令：

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # 或者 "python setup.py develop"
```

注意:

1. 当在 windows 下训练和测试模型时，请确保路径下所有的'\\' 被替换成 '/'，
   在 python 代码里可以使用`.replace('\\', '/')`处理路径的字符串
2. `version+git_hash` 也将被保存进 meta 训练模型里，即0.5.0+c415a2e
3. 当 MMsegmentation 以 `dev` 模式被安装时，本地对代码的修改将不需要重新安装即可产生作用
4. 如果您想使用 `opencv-python-headless` 替换 `opencv-python`，您可以在安装 MMCV 前安装它
5. 一些依赖项是可选的。简单的运行 `pip install -e .` 将仅安装最必要的一些依赖。为了使用可选的依赖项如`cityscapessripts`，
   要么手动使用 `pip install -r requirements/optional.txt` 安装，要么专门从pip下安装(即 `pip install -e .[optional]`，
   其中选项可设置为 `all`, `tests`, `build`, 和 `optional`)

### 完整的安装脚本

#### Linux

这里便是一个完整安装 MMSegmentation 的脚本，使用 conda 并链接了数据集的路径（以您的数据集路径为 $DATA_ROOT 来安装）。

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
pip install mmcv-full==latest+torch1.5.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # 或者 "python setup.py develop"

mkdir data
ln -s $DATA_ROOT data
```

#### Windows (有风险)

这里便是一个完整安装 MMSegmentation 的脚本，使用 conda 并链接了数据集的路径（以您的数据集路径为 %DATA_ROOT% 来安装）。
注意：它必须是一个绝对路径。

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
set PATH=full\path\to\your\cpp\compiler;%PATH%
pip install mmcv

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # 或者 "python setup.py develop"

mklink /D data %DATA_ROOT%
```

#### 使用多版本 MMSegmentation 进行开发

训练和测试脚本已经修改了 `PYTHONPATH` 来确保使用当前路径的MMSegmentation。

为了使用当前环境默认安装的 MMSegmentation 而不是正在工作的 MMSegmentation，您可以在那些脚本里移除下面的内容：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## 验证

为了验证 MMSegmentation 和它所需要的环境是否正确安装，我们可以使用样例 python 代码来初始化一个 segmentor 并推理一张 demo 图像。

```python
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# 从一个 config 配置文件和 checkpoint 文件里创建分割模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# 测试一张样例图片并得到结果
img = 'test.jpg'  # 或者 img = mmcv.imread(img), 这将只加载图像一次．
result = inference_segmentor(model, img)
# 在新的窗口里可视化结果
model.show_result(img, result, show=True)
# 或者保存图片文件的可视化结果
# 您可以改变 segmentation map 的不透明度(opacity)，在(0, 1]之间。
model.show_result(img, result, out_file='result.jpg', opacity=0.5)

# 测试一个视频并得到分割结果
video = mmcv.VideoReader('video.mp4')
for frame in video:
   result = inference_segmentor(model, frame)
   model.show_result(frame, result, wait_time=1)
```

当您完成 MMSegmentation 的安装时，上述代码应该可以成功运行。

我们还提供一个 demo 脚本去可视化单张图片。

```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${DEVICE_NAME}] [--palette-thr ${PALETTE}]
```

样例：

```shell
python demo/image_demo.py demo/demo.jpg configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
    checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --palette cityscapes
```

推理的 demo 文档可在此查询：[demo/inference_demo.ipynb](../demo/inference_demo.ipynb) 。
