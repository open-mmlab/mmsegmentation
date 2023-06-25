# Chest Image Dataset for Pneumothorax Segmentation

## Description

This project supports **`Chest Image Dataset for Pneumothorax Segmentation`**, which can be downloaded from [here](https://tianchi.aliyun.com/dataset/83075).

### Dataset Overview

Pneumothorax can be caused by a blunt chest injury, damage from underlying lung disease, or most horrifying—it may occur for no obvious reason at all. On some occasions, a collapsed lung can be a life-threatening event.
Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. An accurate AI algorithm to detect pneumothorax would be useful in a lot of clinical scenarios. AI could be used to triage chest radiographs for priority interpretation, or to provide a more confident diagnosis for non-radiologists.

The dataset is provided by the Society for Imaging Informatics in Medicine(SIIM), American College of Radiology (ACR),Society of Thoracic Radiology (STR) and MD.ai. You can develop a model to classify (and if present, segment) pneumothorax from a set of chest radiographic images. If successful, you could aid in the early recognition of pneumothoraces and save lives.

### Original Statistic Information

| Dataset name                                                          | Anatomical region | Task type    | Modality | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                            |
| --------------------------------------------------------------------- | ----------------- | ------------ | -------- | ------------ | --------------------- | ---------------------- | ------------ | ------------------------------------------------------------------ |
| [pneumothorax segmentation](https://tianchi.aliyun.com/dataset/83075) | thorax            | segmentation | x_ray    | 2            | 12089/-/3205          | yes/-/no               | -            | [CC-BY-SA-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

|    Class Name     | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :---------------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|      normal       |   12089    |   99.75    |    -     |    -     |     -     |     -     |
| pneumothorax area |    2669    |    0.25    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/x_ray/chest_image_pneum/chest_image_pneum_dataset.png)

### Prerequisites

- Python v3.8
- PyTorch v1.10.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `chest_image_pneum/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://tianchi.aliyun.com/dataset/83075) and decompress data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set can't be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── x_ray
  │   │   │   │   ├── chest_image_pneum
  │   │   │   │   │   ├── configs
  │   │   │   │   │   ├── datasets
  │   │   │   │   │   ├── tools
  │   │   │   │   │   ├── data
  │   │   │   │   │   │   ├── train.txt
  │   │   │   │   │   │   ├── test.txt
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

|    Class Name     | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :---------------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|      normal       |    9637    |   99.75    |   2410   |  99.74   |     -     |     -     |
| pneumothorax area |    2137    |    0.25    |   532    |   0.26   |     -     |     -     |

### Training commands

Train models on a single server with one GPU.

```shell
mim train mmseg ./configs/${CONFIG_FILE}
```

### Testing commands

Test models on a single server with one GPU.

```shell
mim test mmseg ./configs/${CONFIG_FILE}  --checkpoint ${CHECKPOINT_PATH}
```

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/configs/fcn#results-and-models)

You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

## Results

### Bactteria detection with darkfield microscopy

|     Method      | Backbone | Crop Size |   lr   | mIoU | mDice |                                         config                                         |         download         |
| :-------------: | :------: | :-------: | :----: | :--: | :---: | :------------------------------------------------------------------------------------: | :----------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  |  -   |   -   |  [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_chest-image-pneum-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  |  -   |   -   | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_chest-image-pneum-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 |  -   |   -   | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_chest-image-pneum-512x512.py) | [model](<>) \| [log](<>) |

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
