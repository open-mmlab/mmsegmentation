## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+
- PyTorch 1.3 or higher
- CUDA 9.2 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.2/10.0/10.1
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC(G++): 4.9/5.3/5.4/7.3

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

d. Install build requirements and then install mmsegmentation.
Please refer to [MMCV](https://mmcv.readthedocs.io/en/latest/) for other versions.

```shell
pip install mmcv==1.0rc0+torch1.5.0+cu101 -f http://8.210.27.39:9000/ --trusted-host 8.210.27.39
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

Note:

1. The git commit id will be written to the version number with step *d*, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step *d* each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

2. Following the above instructions, mmsegmentation is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements.
To use optional dependencies like `cityscapessripts`  either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.


### A from-scratch setup script

Here is a full script for setting up mmsegmentation with conda and link the dataset path (supposing that your dataset path is $DATA_ROOT).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch=1.5.0 torchvision cudatoolkit=10.1 -c pytorch
git clone http://github.com/open-mmlab/mmsegmentation
cd mmsegmentation
pip install mmcv==1.0rc0+torch1.5.0+cu101 -f http://8.210.27.39:9000/ --trusted-host 8.210.27.39
pip install -r requirements/build.txt
pip install -v -e .

mkdir data
ln -s $DATA_ROOT data
```
