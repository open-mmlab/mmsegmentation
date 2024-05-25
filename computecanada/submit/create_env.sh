set -e
# deactivate
module purge
module load  StdEnv/2020 python/3.10.2
module load gcc/9.3.0 opencv/4.8.0 cuda/11.7
echo "loading module done"

echo "Creating new virtualenv"

virtualenv ~/$1
source ~/$1/bin/activate

echo "Activating virtual env"

cd ../../

pip install tqdm
pip install sklearn
pip install jupyterlab
pip install ipywidgets
pip install icecream
pip install wandb
pip install matplotlib
pip install numpy
pip install torch
pip install torchvision
pip install xarray
pip install h5netcdf
# pip install mim
pip install torchmetrics
pip install ftfy
pip install regex
pip install mmengine>=0.8.3
pip install mmcv
pip install -v -e .

# mmwhale_dir=$(pwd)

# pip install --no-index --upgrade pip

# pip install opencv-python-headless

# pip install numpy
# pip install matplotlib
# pip install torch==1.13.1
# pip install torchvision
# pip install tqdm
# pip install sklearn
# # pip install # ipywidgets==8.0.2
# pip install jupyterlab
# pip install ipywidgets
# pip install icecream
# pip install wandb

# # mim installation
# pip install -U openmim
# mim install mmengine


# # build mmcv from source-- NEED TO HAVE GPU FOR THIS
# git clone https://github.com/open-mmlab/mmcv.git
# cd ../mmcv
# MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -v -e .

# # install mmcv from pip
# # MMCV_WITH_OPS=1 FORCE_CUDA=1  mim install mmcv --no-deps

# # install mmde
# cd $mmwhale_dir
# pip install -v -e .

# # pip uninstall opencv-python
# # pip uninstall opencv-python-headless
# # pip install opencv-python-headless
