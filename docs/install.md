## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.3 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

### Install mmsegmentation

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).
Here we use PyTorch 1.5.0 and CUDA 10.1.
You may also switch to other version by specifying version number.

```shell
conda install pytorch=1.5.0 torchvision cudatoolkit=10.1 -c pytorch
```

c. Clone the mmsegmentation repository.

```shell
git clone http://github.com/open-mmlab/mmsegmentation
cd mmsegmentation
```

d. Install [MMCV](https://mmcv.readthedocs.io/en/latest/).
Either *mmcv* or *mmcv-full* is compatible with MMSegmentation, but for methods like CCNet and PSANet, CUDA ops in *mmcv-full* is required

The pre-build *mmcv-full* could be installed by running: (available versions could be found [here](https://mmcv.readthedocs.io/en/latest/#install-with-pip))
```
pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
```

Optionally, you could also install lite version by running:
```
pip install mmcv
```
or build full version from source:
```
pip install mmcv-full
```

e. Install build requirements and then install MMSegmentation.

```shell
pip install -r requirements/build.txt  # or "pip install -r requirements.txt" for everything.
pip install -e .  # or "python setup.py develop"
```

Note:

1. The git commit id will be written to the version number with step *e*, e.g. 0.5.0+c415a2e. The version will also be saved in trained models.
It is recommended that you run step *e* each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

2. Following the above instructions, mmsegmentation is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. Some dependencies are optional. Simply running `pip install -e .` will only install the minimum runtime requirements.
To use optional dependencies like `cityscapessripts`  either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.


### A from-scratch setup script

Here is a full script for setting up mmsegmentation with conda and link the dataset path (supposing that your dataset path is $DATA_ROOT).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch=1.5.0 torchvision cudatoolkit=10.1 -c pytorch
git clone http://github.com/open-mmlab/mmsegmentation
cd mmsegmentation
pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
pip install -r requirements/build.txt
pip install -e .

mkdir data
ln -s $DATA_ROOT data
```
