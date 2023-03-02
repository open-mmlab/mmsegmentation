# PanNuke

## Description

This project support **`PanNuke `**, and the dataset used in this project can be downloaded from [here](https://academictorrents.com/details/99f2c7b57b95500711e33f2ee4d14c9fd7c7366c).

### Dataset Overview

Semi automatically generated nuclei instance segmentation and classification dataset with exhaustive nuclei labels across 19 different tissue types. The dataset consists of 481 visual fields, of which 312 are randomly sampled from more than 20K whole slide images at different magnifications, from multiple data sources. In total the dataset contains 205,343 labeled nuclei, each with an instance segmentation mask. Models trained on pannuke can aid in whole slide image tissue type segmentation, and generalise to new tissues. PanNuke demonstrates one of the first successfully semi-automatically generated datasets.

### Statistic Information

| dataset name                                                                             | anatomical region | task type    | modality       | num. classes | train/val/test images | release date | License                                                         |
| ---------------------------------------------------------------------------------------- | ----------------- | ------------ | -------------- | ------------ | --------------------- | ------------ | --------------------------------------------------------------- |
| [Pannuke](https://academictorrents.com/details/99f2c7b57b95500711e33f2ee4d14c9fd7c7366c) | full_body         | segmentation | histopathology | 6            | 7901/-/-              | 2019         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| class name                | pixel percentage | Num. images in this category |
| ------------------------- | ---------------- | ---------------------------- |
| background                | 83.32            | 7901                         |
| neoplastic                | 8.64             | 4190                         |
| non-neoplastic epithelial | 1.77             | 4126                         |
| inflammatory              | 3.73             | 6137                         |
| connective                | 0.07             | 232                          |
| dead                      | 2.47             | 1528                         |

### Visualization

![pannuke](https://github.com/uni-medical/medical-datasets-visualization/blob/main/2d/semantic_seg/histopathology/pannuke/pannuke_dataset.png?raw=true)

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- pillow(PIL) 9.3.0
- scikit-learn(sklearn) 1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of PYTHONPATH, which should point to the project's directory so that Python can locate the module files. In pannuke/ root directory, run the following line to add the current directory to PYTHONPATH:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://academictorrents.com/details/99f2c7b57b95500711e33f2ee4d14c9fd7c7366c) and decompression data to path 'data/'.
- run script `"python tools/prepare_dataset.py"` to split dataset and change folder structure as below.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── histopathology
  │   │   │   │   ├── pannuke
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

### PanNuke

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                        config                                                                                         |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/histopathology/pannuke/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_pannuke-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/histopathology/pannuke/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_pannuke-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/histopathology/pannuke/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_pannuke-512x512.py) |

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
