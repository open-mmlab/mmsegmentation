# Retinal Images vessel Tree Extraction (RITE)

## Description

This project support **`Retinal Images vessel Tree Extraction (RITE) `**, and the dataset used in this project can be downloaded from [here](https://opendatalab.com/RITE).

### Dataset Overview

The RITE (Retinal Images vessel Tree Extraction) is a database that enables comparative studies on segmentation or classification of arteries and veins on retinal fundus images, which is established based on the public available DRIVE database (Digital Retinal Images for Vessel Extraction). RITE contains 40 sets of images, equally separated into a training subset and a test subset, the same as DRIVE. The two subsets are built from the corresponding two subsets in DRIVE. For each set, there is a fundus photograph, a vessel reference standard. The fundus photograph is inherited from DRIVE. For the training set, the vessel reference standard is a modified version of 1st_manual from DRIVE. For the test set, the vessel reference standard is 2nd_manual from DRIVE.

### Statistic Information

| dataset name                         | anatomical region | task type    | modality           | num. classes | train/val/test images | release date | License                                                         |
| ------------------------------------ | ----------------- | ------------ | ------------------ | ------------ | --------------------- | ------------ | --------------------------------------------------------------- |
| [Rite](https://opendatalab.com/RITE) | head_and_neck     | segmentation | fundus_photography | 2            | 20/-/20               | 2013         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| class name | pixel percentage | Num. images in this category |
| ---------- | ---------------- | ---------------------------- |
| background | 91.60            | 40                           |
| vessel     | 8.40             | 40                           |

### Visualization

![rite](https://github.com/uni-medical/medical-datasets-visualization/blob/main/2d/semantic_seg/fundus_photography/rite/rite_dataset.png?raw=true)

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- pillow(PIL) 9.3.0
- scikit-learn(sklearn) 1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of PYTHONPATH, which should point to the project's directory so that Python can locate the module files. In rite/ root directory, run the following line to add the current directory to PYTHONPATH:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://opendatalab.com/RITE) and decompression data to path 'data/'.
- run script `"python tools/prepare_dataset.py"` to split dataset and change folder structure as below.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── fundus_photography
  │   │   │   │   ├── rite
  │   │   │   │   │   ├── configs
  │   │   │   │   │   ├── datasets
  │   │   │   │   │   ├── tools
  │   │   │   │   │   ├── data
  │   │   │   │   │   │   ├── images
  │   │   │   │   │   │   │   ├── train
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png
  │   │   │   │   │   │   │   ├── val
  │   │   │   │   |   │   │   │   ├── yyy.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── yyy.png
  │   │   │   │   │   │   ├── masks
  │   │   │   │   │   │   │   ├── train
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png
  │   │   │   │   │   │   │   ├── val
  │   │   │   │   |   │   │   │   ├── yyy.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── yyy.png
```

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

### Retinal Images Vessel Tree Extraction

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                       config                                                                                        |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/rite/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_rite-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/rite/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_rite-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/rite/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_rite-512x512.py) |

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
