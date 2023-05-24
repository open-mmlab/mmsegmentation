# Get started: Install and Run MMSeg

## Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMSegmentation works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.5+.

**Note:**
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](##installation). Otherwise, you can follow these steps for the preparation.

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## Installation

We recommend that users follow our best practices to install MMSegmentation. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

### Best Practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 1.** Install MMSegmentation.

Case a: If you develop and run mmseg directly, install it from source:

```shell
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case b: If you use mmsegmentation as a dependency or third-party package, install it with pip:

```shell
pip install "mmsegmentation>=1.0.0"
```

### Verify the installation

To verify whether MMSegmentation is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py` and `pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth` in your current folder.

**Step 2.** Verify the inference demo.

Option (a). If you install mmsegmentation from source, just run the following command.

```shell
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```

You will see a new image `result.jpg` on your current folder, where segmentation masks are covered on all objects.

Option (b). If you install mmsegmentation with pip, open you python interpreter and copy&paste the following codes.

```python
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo/demo.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_model(model, img)
# visualize the results in a new window
show_result_pyplot(model, img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)
# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
   result = inference_segmentor(model, frame)
   show_result_pyplot(model, result, wait_time=1)
```

You can modify the code above to test a single image or a video, both of these options can verify that the installation was successful.

### Customize Installation

#### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

**Note:**
Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.

#### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv==2.0.0 built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

#### Install on CPU-only platforms

MMSegmentation can be built for CPU only environment. In CPU mode you can train (requires MMCV version >= 2.0.0), test or inference a model.

#### Install on Google Colab

[Google Colab](https://research.google.com/) usually has PyTorch installed,
thus we only need to install MMCV and MMSegmentation with the following commands.

**Step 1.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
!pip3 install openmim
!mim install mmengine
!mim install "mmcv>=2.0.0"
```

**Step 2.** Install MMSegmentation from the source.

```shell
!git clone https://github.com/open-mmlab/mmsegmentation.git
%cd mmsegmentation
!git checkout main
!pip install -e .
```

**Step 3.** Verification.

```python
import mmseg
print(mmseg.__version__)
# Example output: 1.0.0
```

**Note:**
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.

### Using MMSegmentation with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmsegmentation/blob/main/docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.11, CUDA 11.3
# If you prefer other versions, just modified the Dockerfile
docker build -t mmsegmentation docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmsegmentation/data mmsegmentation
```

### Optional Dependencies

#### Install GDAL

[GDAL](https://gdal.org/) is a translator library for raster and vector geospatial data formats. Install GDAL to read complex formats and extremely large remote sensing images.

```shell
conda install GDAL
```

## Trouble shooting

If you have some issues during the installation, please first view the [FAQ](notes/faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmsegmentation/issues/new/choose) on GitHub if no solution is found.
