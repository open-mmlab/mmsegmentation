# Breast Cancer Semantic Segmentation (BCSS)

## Description

This project support **`Breast Cancer Semantic Segmentation (BCSS) `**, and the dataset used in this project can be downloaded from [here](https://bcsegmentation.grand-challenge.org/).

### Dataset Overview

The BCSS dataset contains over 20,000 segmentation annotations of tissue region from breast cancer images from TCGA. This large-scale dataset was annotated through the collaborative effort of pathologists, pathology residents, and medical students using the Digital Slide Archive.  It enables the generation of highly accurate machine-learning models for tissue segmentation.

### Information Statistics

| Dataset Name                                        | Anatomical Region | Task Type    | Modality       | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                       |
| --------------------------------------------------- | ----------------- | ------------ | -------------- | ------------ | --------------------- | ---------------------- | ------------ | ------------------------------------------------------------- |
| [bcss](https://bcsegmentation.grand-challenge.org/) | throax            | segmentation | histopathology | 6            | 151/-/-               | yes/-/-                | 2019         | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

|  Class Name  | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :----------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|  background  |     49     |   21.75    |    -     |    -     |     -     |     -     |
|    tumor     |    151     |   31.04    |    -     |    -     |     -     |     -     |
|    stroma    |    151     |   28.16    |    -     |    -     |     -     |     -     |
| inflammatory |    126     |    9.44    |    -     |    -     |     -     |     -     |
|   necrosis   |     91     |    5.16    |    -     |    -     |     -     |     -     |
|    other     |    141     |    4.45    |    -     |    -     |     -     |     -     |

### Visualization

![bcss](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/histopathology/bcss/bcss_dataset.png?raw=true)

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- pillow(PIL)
- scikit-learn(sklearn)
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of PYTHONPATH, which should point to the project's directory so that Python can locate the module files. In bcss/ root directory, run the following line to add the current directory to PYTHONPATH:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://bcsegmentation.grand-challenge.org/) and decompression data to path 'data/'.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set cannot be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── histopathology
  │   │   │   │   ├── bcss
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

|  Class Name  | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :----------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|  background  |     40     |   22.13    |    9     |  19.96   |     -     |     -     |
|    tumor     |    120     |   30.71    |    31    |  32.57   |     -     |     -     |
|    stroma    |    120     |   27.83    |    31    |  29.67   |     -     |     -     |
| inflammatory |     98     |    9.7     |    28    |   8.23   |     -     |     -     |
|   necrosis   |     76     |    5.32    |    15    |   4.44   |     -     |     -     |
|    other     |    112     |    4.3     |    29    |   5.13   |     -     |     -     |

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

### Breast Cancer Semantic Segmentation (BCSS)

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                     config                                                                                      |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/histopathology/bcss/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_bcss-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/histopathology/bcss/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_bcss-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/histopathology/bcss/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_bcss-512x512.py) |

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code
  - [x] Basic docstrings & proper citation
  - [ ] Test-time correctness
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
