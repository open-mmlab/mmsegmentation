## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

The compatible MMSegmentation and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

| MMSegmentation version |    MMCV version     |
|:-------------------:|:-------------------:|
| master              | mmcv-full>=1.3.13, <1.4.0 |
| 0.19.0              | mmcv-full>=1.3.13, <1.4.0 |
| 0.18.0              | mmcv-full>=1.3.13, <1.4.0 |
| 0.17.0              | mmcv-full>=1.3.7, <1.4.0 |
| 0.16.0              | mmcv-full>=1.3.7, <1.4.0 |
| 0.15.0              | mmcv-full>=1.3.7, <1.4.0 |
| 0.14.1              | mmcv-full>=1.3.7, <1.4.0 |
| 0.14.0              | mmcv-full>=1.3.1, <1.3.2 |
| 0.13.0              | mmcv-full>=1.3.1, <1.3.2 |
| 0.12.0              | mmcv-full>=1.1.4, <1.3.2 |
| 0.11.0              | mmcv-full>=1.1.4, <1.3.0 |
| 0.10.0              | mmcv-full>=1.1.4, <1.3.0 |
| 0.9.0               | mmcv-full>=1.1.4, <1.3.0 |
| 0.8.0               | mmcv-full>=1.1.4, <1.2.0 |
| 0.7.0               | mmcv-full>=1.1.2, <1.2.0 |
| 0.6.0               | mmcv-full>=1.1.2, <1.2.0 |

:::{note}
You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.
:::

## Installation

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).
Here we use PyTorch 1.6.0 and CUDA 10.1.
You may also switch to other version by specifying the version number.

```shell
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
```

c. Install [MMCV](https://mmcv.readthedocs.io/en/latest/) following the [official instructions](https://mmcv.readthedocs.io/en/latest/#installation).
Either `mmcv` or `mmcv-full` is compatible with MMSegmentation, but for methods like CCNet and PSANet, CUDA ops in `mmcv-full` is required.

**Install mmcv for Linux:**

The pre-build mmcv-full (with PyTorch 1.6 and CUDA 10.1) can be installed by running: (other available versions could be found [here](https://mmcv.readthedocs.io/en/latest/#install-with-pip))

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
```

**Install mmcv for Windows (Experimental):**

For Windows, the installation of MMCV requires native C++ compilers, such as cl.exe. Please add the compiler to %PATH%.

A typical path for cl.exe looks like the following if you have Windows SDK and Visual Studio installed on your computer:

```shell
C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.26.28801\bin\Hostx86\x64
```

Or you should download the cl compiler from web and then set up the path.

Then, clone mmcv from github and install mmcv via pip:

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -e .
```

Or simply:

```shell
pip install mmcv
```

Currently, mmcv-full is not supported on Windows.

d. Install MMSegmentation.

```shell
pip install mmsegmentation # install the latest release
```

or

```shell
pip install git+https://github.com/open-mmlab/mmsegmentation.git # install the master branch
```

Instead, if you would like to install MMSegmentation in `dev` mode, run following

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # or "python setup.py develop"
```

:::{note}

1. When training or testing models on Windows, please ensure that all the '\\' in paths are replaced with '/'. Add .replace('\\', '/') to your python code wherever path strings occur.
2. The `version+git_hash` will also be saved in trained models meta, e.g. 0.5.0+c415a2e.
3. When MMsegmentation is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it.
4. If you would like to use `opencv-python-headless` instead of `opencv-python`,
   you can install it before installing MMCV.
5. Some dependencies are optional. Simply running `pip install -e .` will only install the minimum runtime requirements.
   To use optional dependencies like `cityscapessripts`  either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.
:::

### A from-scratch setup script

#### Linux

Here is a full script for setting up mmsegmentation with conda and link the dataset path (supposing that your dataset path is $DATA_ROOT).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # or "python setup.py develop"

mkdir data
ln -s $DATA_ROOT data
```

#### Windows(Experimental)

Here is a full script for setting up mmsegmentation with conda and link the dataset path (supposing that your dataset path is
%DATA_ROOT%. Notice: It must be an absolute path).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
set PATH=full\path\to\your\cpp\compiler;%PATH%
pip install mmcv

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # or "python setup.py develop"

mklink /D data %DATA_ROOT%
```

#### Developing with multiple MMSegmentation versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMSegmentation in the current directory.

To use the default MMSegmentation installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## Verification

To verify whether MMSegmentation and the required environment are installed correctly, we can run sample python codes to initialize a segmentor and inference a demo image:

```python
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)
# visualize the results in a new window
model.show_result(img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
model.show_result(img, result, out_file='result.jpg', opacity=0.5)

# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
   result = inference_segmentor(model, frame)
   model.show_result(frame, result, wait_time=1)
```

The above code is supposed to run successfully upon you finish the installation.

We also provide a demo script to test a single image.

```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${DEVICE_NAME}] [--palette-thr ${PALETTE}]
```

Examples:

```shell
python demo/image_demo.py demo/demo.jpg configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
    checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --palette cityscapes
```

A notebook demo can be found in [demo/inference_demo.ipynb](../demo/inference_demo.ipynb).
