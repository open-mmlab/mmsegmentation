## 依赖

- Linux or macOS （Windows系统处于试验性支持阶段）
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ （如果您基于源文件编译 PyTorch，CUDA 9.0也是兼容的）
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

兼容的 MMSegmentation 和 MMCV 版本如下所示，请安装对应版本的 MMCV 以避免出现兼容问题。

| MMSegmentation 版本 |    MMCV 版本     |
|:-------------------:|:-------------------:|
| master              | mmcv-full>=1.3.7, <1.4.0 |
| 0.15.0              | mmcv-full>=1.3.7, <1.4.0 |
| 0.14.1              | mmcv-full>=1.3.7, <1.4.0 |
| 0.14.0              | mmcv-full>=1.3.1, <1.4.0 |
| 0.13.0              | mmcv-full>=1.3.1, <1.4.0 |
| 0.12.0              | mmcv-full>=1.1.4, <1.4.0 |
| 0.11.0              | mmcv-full>=1.1.4, <1.3.0 |
| 0.10.0              | mmcv-full>=1.1.4, <1.3.0 |
| 0.9.0               | mmcv-full>=1.1.4, <1.3.0 |
| 0.8.0               | mmcv-full>=1.1.4, <1.2.0 |
| 0.7.0               | mmcv-full>=1.1.2, <1.2.0 |
| 0.6.0               | mmcv-full>=1.1.2, <1.2.0 |

注意：如果您安装了 MMCV，您首先需要运行 `pip uninstall mmcv` 进行卸载。如果 mmcv 和 mmcv-full 并存于环境，会出现 `ModuleNotFoundError` 错误。

## 安装

1. 创建一个 conda 虚拟环境并激活它。

   ```shell
   conda create -n open-mmlab python=3.7 -y
   conda activate open-mmlab
   ```

2. 按照 [官方教程](https://pytorch.org/) 安装 PyTorch 和 torchvision。这里以 PyTorch1.6.0 和 CUDA10.1 为例。您也可以通过指定版本号安装其他版本。

   ```shell
   conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
   ```

3. 按照 [官方教程](https://mmcv.readthedocs.io/en/latest/#installation) 安装 [MMCV](https://mmcv.readthedocs.io/en/latest/) 。`mmcv` 或 `mmcv-full` 都与 MMSegmentation 兼容，但对于像 CCNet 和 PSANet这样的方法，需要 `mmcv-full` 中的 CUDA 算子支持。

   - **在 Linux 下安装 mmcv：**

     通过运行

     ```shell
     pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
     ```

     可以安装预编译的 mmcv-full （PyTorch 1.6 和 CUDA 10.1） 。其他 PyTorch 和 CUDA 版本的 MMCV 的安装请参照[这里](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。

   - **在 Windows 下安装 mmcv （试验性）：**

     对于 Windows，MMCV 的安装需要本地 C++ 编译工具，例如 cl.exe。 请将编译器路径添加到 %PATH%。

     如果您已经在电脑上安装了Windows SDK 和 Visual Studio，cl.exe 的典型路径如下：

     ```shell
     C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.26.28801\bin\Hostx86\x64
     ```

     或者您需要从网上下载 cl 编译器，然后设置好路径。

     随后，从 github 克隆 mmcv 并通过 pip 安装：

     ```shell
     git clone https://github.com/open-mmlab/mmcv.git
     cd mmcv
     pip install -e .
     ```

     或直接：

     ```shell
     pip install mmcv
     ```

     目前，在 Windows 上 mmcv-full 并不完全支持。

4. 安装 MMSegmentation。

   ```shell
   pip install mmsegmentation # 安装最新版本
   ```

   或者

   ```shell
   pip install git+https://github.com/open-mmlab/mmsegmentation.git # 安装 master 分支
   ```

   此外，如果您想安装 `dev` 模式的 MMSegmentation，运行如下命令：

   ```shell
   git clone https://github.com/open-mmlab/mmsegmentation.git
   cd mmsegmentation
   pip install -e .  # 或者 "python setup.py develop"
   ```

   注意：

   - 当在 windows 下训练和测试模型时，请确保所有路径中的 '\\' 都被替换成 '/'。在 python 代码里，凡是出现路径字符串的地方，都可以通过添加 `.replace('\\', '/')` 进行处理。
   - `version+git_hash` 也将被保存在训练好的模型的 meta 里，例如 0.5.0+c415a2e。
   - 当 MMsegmentation 以 `dev` 模式被安装时，本地对代码的修改将不需要重新安装即可产生作用。
   - 如果您想使用 `opencv-python-headless` 而不是 `opencv-python`，您可以在安装 MMCV 之前安装它。
   - 一些依赖项是可选的。简单地运行 `pip install -e .` 将仅安装最必要的一些依赖。为了使用可选的依赖项，例如`cityscapessripts`，可以用 `pip install -r requirements/optional.txt` 手动安装，或者在调用pip时指定所需的额外参数（例如 `pip install -e .[optional]`， 其中选项可设置为 `all`, `tests`, `build`, 和 `optional`）。

### 完整的安装脚本

#### Linux

下面是一个使用 conda 完整安装 MMSegmentation 并链接数据集路径（假设您的数据集路径为 $DATA_ROOT ）的脚本。

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # 或者 "python setup.py develop"

mkdir data
ln -s $DATA_ROOT data
```

#### Windows（试验性）

下面是一个使用 conda 完整安装 MMSegmentation 并链接数据集路径（假设您的数据集路径为 $DATA_ROOT ，$DATA_ROOT 必须为一个绝对路径）的脚本。

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

训练和测试脚本已经修改了 `PYTHONPATH` 以确保脚本使用当前工作目录下的MMSegmentation。

要使用安装在环境中的 MMSegmentation 而不是工作区中的，您可以在训练和测试脚本中移除下面的内容：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## 验证

为了验证 MMSegmentation 和它所需要的环境是否正确安装，我们可以运行示例的 python 代码来初始化一个 segmentor 并推理一张 demo 图像。

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

当您完成 MMSegmentation 的安装时，上述代码可以成功运行。

我们还提供一个 demo 脚本去可视化单张图片

```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${DEVICE_NAME}] [--palette-thr ${PALETTE}]
```

样例：

```shell
python demo/image_demo.py demo/demo.jpg configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
    checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --palette cityscapes
```

notebook 格式的 demo 示例路径：[demo/inference_demo.ipynb](../demo/inference_demo.ipynb)。

