# ISIC-2016 Task1

## Description

This project support **`ISIC-2016 Task1 `**, and the dataset used in this project can be downloaded from [here](https://challenge.isic-archive.com/data/#2016).

### Dataset Overview

The overarching goal of the challenge is to develop image analysis tools to enable the automated diagnosis of melanoma from dermoscopic images.

This challenge provides training data (~900 images) for participants to engage in all 3 components of lesion image analysis. A separate test dataset (~350 images) will be provided for participants to generate and submit automated results.

### Original Statistic Information

| Dataset name                                                     | Anatomical region | Task type    | Modality   | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                                |
| ---------------------------------------------------------------- | ----------------- | ------------ | ---------- | ------------ | --------------------- | ---------------------- | ------------ | ---------------------------------------------------------------------- |
| [ISIC-2016 Task1](https://challenge.isic-archive.com/data/#2016) | full body         | segmentation | dermoscopy | 2            | 900/-/379-            | yes/-/yes              | 2016         | [CC-0](https://creativecommons.org/share-your-work/public-domain/cc0/) |

| Class Name  | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :---------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background  |    900     |   82.08    |    -     |    -     |    379    |   81.98   |
| skin lesion |    900     |   17.92    |    -     |    -     |    379    |   18.02   |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/dermoscopy/isic2016_task1/isic2016_task1.png)

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- pillow(PIL) 9.3.0
- scikit-learn(sklearn) 1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of PYTHONPATH, which should point to the project's directory so that Python can locate the module files. In isic2016_task1/ root directory, run the following line to add the current directory to PYTHONPATH:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://challenge.isic-archive.com/data/#2016) and decompression data to path 'data/'.
- run script `"python tools/prepare_dataset.py"` to split dataset and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt` and `test.txt`. If the label of official validation set and test set can't be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── dermoscopy
  │   │   │   │   ├── isic2016_task1
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
  │   │   │   │   |   │   │   │   └── xxx.png
  │   │   │   │   │   │   │   ├── test
  │   │   │   │   |   │   │   │   ├── yyy.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── yyy.png
  │   │   │   │   │   │   ├── masks
  │   │   │   │   │   │   │   ├── train
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png
  │   │   │   │   │   │   │   ├── test
  │   │   │   │   |   │   │   │   ├── yyy.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── yyy.png
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

### ISIC-2016 Task1

|     Method      | Backbone | Crop Size |   lr   | mIoU | mDice |                                                                                             config                                                                                              |
| :-------------: | :------: | :-------: | :----: | :--: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  |  -   |   -   |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/dermoscopy/isic2016_task1/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_isic2016-task1-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  |  -   |   -   | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/dermoscopy/isic2016_task1/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_isic2016-task1-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 |  -   |   -   | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/dermoscopy/isic2016_task1/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_isic2016-task1-512x512.py) |

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
