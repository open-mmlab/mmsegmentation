## Requirements

- Linux or Windows(Experimental)
- Python 3.6+
- PyTorch 1.3 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

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

The pre-build mmcv-full (with PyTorch 1.5 and CUDA 10.1) can be installed by running: (other available versions could be found [here](https://mmcv.readthedocs.io/en/latest/#install-with-pip))

```shell
pip install mmcv-full==latest+torch1.5.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
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

Note:

1. When training or testing models on Windows, please ensure that all the '\\' in paths are replaced with '/'. Add .replace('\\', '/') to your python code wherever path strings occur.
2. The `version+git_hash` will also be saved in trained models meta, e.g. 0.5.0+c415a2e.
3. When MMsegmentation is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it.
4. If you would like to use `opencv-python-headless` instead of `opencv-python`,
   you can install it before installing MMCV.
5. Some dependencies are optional. Simply running `pip install -e .` will only install the minimum runtime requirements.
   To use optional dependencies like `cityscapessripts`  either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

## A from-scratch setup script

### Linux

Here is a full script for setting up mmsegmentation with conda and link the dataset path (supposing that your dataset path is $DATA_ROOT).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
pip install mmcv-full==latest+torch1.5.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # or "python setup.py develop"

mkdir data
ln -s $DATA_ROOT data
```

### Windows(Experimental)

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
