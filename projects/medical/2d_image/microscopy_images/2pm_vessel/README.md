# 2-PM Vessel Dataset

## Description

This project supports **`2-PM Vessel Dataset`**, which can be downloaded from [here](https://opendatalab.org.cn/2-PM_Vessel_Dataset).

### Dataset Overview

An open-source volumetric brain vasculature dataset obtained with two-photon microscopy at Focused Ultrasound Lab, at Sunnybrook Research Institute (affiliated with University of Toronto by Dr. Alison Burgess, Charissa Poon and Marc Santos).

The dataset contains a total of 12 volumetric stacks consisting images of mouse brain vasculature and tumor vasculature.

### Information Statistics

| Dataset Name                                                 | Anatomical Region | Task Type    | Modality          | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                       |
| ------------------------------------------------------------ | ----------------- | ------------ | ----------------- | ------------ | --------------------- | ---------------------- | ------------ | ------------------------------------------------------------- |
| [2pm_vessel](https://opendatalab.org.cn/2-PM_Vessel_Dataset) | vessel            | segmentation | microscopy_images | 2            | 216/-/-               | yes/-/-                | 2021         | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    216     |   85.78    |    -     |    -     |     -     |     -     |
|   vessel   |    180     |   14.22    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![2pmv](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/histopathology/2pm_vessel/2pm_vessel_dataset.png?raw=true)

### Dataset Citation

```
@article{teikari2016deep,
	title={Deep learning convolutional networks for multiphoton microscopy vasculature segmentation},
	author={Teikari, Petteri and Santos, Marc and Poon, Charissa and Hynynen, Kullervo},
	journal={arXiv preprint arXiv:1606.02382},
	year={2016}
}
```

### Prerequisites

- Python v3.8
- PyTorch v1.10.0
- pillow(PIL) v9.3.0
- scikit-learn(sklearn) v1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `2pm_vessel/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- download dataset from [here](https://opendatalab.org.cn/2-PM_Vessel_Dataset) and decompress data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set can't be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```shell
mkdir data & cd data
pip install opendatalab
odl get    2-PM_Vessel_Dataset
cd ..
python tools/prepare_dataset.py
python ../../tools/split_seg_dataset.py
```

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── microscopy_images
  │   │   │   │   ├── 2pm_vessel
  │   │   │   │   │   ├── configs
  │   │   │   │   │   ├── datasets
  │   │   │   │   │   ├── tools
  │   │   │   │   │   ├── data
  │   │   │   │   │   │   ├── train.txt
  │   │   │   │   │   │   ├── val.txt
  │   │   │   │   │   │   ├── images
  │   │   │   │   │   │   │   ├── train
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png
  │   │   │   │   │   │   ├── masks
  │   │   │   │   │   │   │   ├── train
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png

```

### Divided Dataset Information

***Note: The table information below is divided by ourselves.***

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    172     |   85.88    |    44    |   85.4   |     -     |     -     |
|   vessel   |    142     |   14.12    |    38    |   14.6   |     -     |     -     |

### Training commands

To train models on a single server with one GPU. (default)

```shell
mim train mmseg ./configs/${CONFIG_FILE}
```

### Testing commands

To test models on a single server with one GPU. (default)

```shell
mim test mmseg ./configs/${CONFIG_FILE}  --checkpoint ${CHECKPOINT_PATH}
```

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/configs/fcn#results-and-models)

You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code
  - [x] Basic docstrings & proper citation
  - [ ] Test-time correctness
  - [x] A full README

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings
  - [ ] Unit tests
  - [ ] Code polishing
  - [ ] Metafile.yml

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
