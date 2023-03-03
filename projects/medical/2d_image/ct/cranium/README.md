# Cranium

## Description

This project support **`Cranium `**, and the dataset used in this project can be downloaded from [here](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images).

### Dataset Overview

This dataset consists of head CT (Computed Thomography) images in jpg format. There are 2500 brain window images and 2500 bone window images, for 82 patients. There are approximately 30 image slices per patient. 318 images have associated intracranial image masks. Also included are csv files containing hemorrhage diagnosis data and patient data.
This is version 1.0.0 of this dataset. A full description of this dataset as well as updated versions can be found here:
https://physionet.org/content/ct-ich/1.0.0/

### Statistic Information

| Dataset Name                                                                        | Anatomical Region | Task Type    | Modality | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                   |
| ----------------------------------------------------------------------------------- | ----------------- | ------------ | -------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------- |
| [Cranium](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images) | head_and_neck     | segmentation | ct       | 2            | 2501/-/-              | yes/-/-                | 2020         | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    2501    |   99.93    |    -     |    -     |     -     |     -     |
| hemorrhage |    318     |    0.07    |    -     |    -     |     -     |     -     |

Note:

- `pct` means percentage of pixels in this category in all pixels.

### Visualization

![cranium](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/ct/cranium/cranium_dataset.png?raw=true)

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- pillow(PIL) 9.3.0
- scikit-learn(sklearn) 1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of PYTHONPATH, which should point to the project's directory so that Python can locate the module files. In cranium/ root directory, run the following line to add the current directory to PYTHONPATH:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images) and decompression data to path 'data/'.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set cannot be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── ct
  │   │   │   │   ├── cranium
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
| background |    2000    |   99.93    |   501    |  99.92   |     -     |     -     |
| hemorrhage |    260     |    0.07    |   260    |   0.08   |     -     |     -     |

### Training commands

```shell
mim train mmseg ./configs/${CONFIG_PATH}
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```shell
mim train mmseg ./configs/${CONFIG_PATH}  --launcher pytorch --gpus 8
```

### Testing commands

```shell
mim test mmseg ./configs/${CONFIG_PATH}  --checkpoint ${CHECKPOINT_PATH}
```

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/configs/fcn#results-and-models)

You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

## Results

### Cranium

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                  config                                                                                   |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/ct/cranium/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_cranium-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/ct/cranium/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_cranium-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/ct/cranium/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_cranium-512x512.py) |

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code
  - [x] Basic docstrings & proper citation
  - [x] Test-time correctness
  - [x] A full README

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings
  - [ ] Unit tests
  - [ ] Code polishing
  - [ ] Metafile.yml

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
