# Japanese Society of Radiological Technology Dataset (JSRT)

## Description

This project support **`Japanese Society of Radiological Technology Dataset (JSRT) `**, and the dataset used in this project can be downloaded from [here](http://db.jsrt.or.jp/eng.php).

### Dataset Overview

The chest X-ray image is recorded in png file form, with 199 cases for training and 48 cases for test. The label image is a binary image of the lung field area. The definition and determination of the lung field regions in the label data are not medically supervised, so there is no medical basis for them.

### Information Statistics

| dataset_name                         | anatomical region | task type    | modality | number of categories | train/val/test image | release date | License                                                       |
| ------------------------------------ | ----------------- | ------------ | -------- | -------------------- | -------------------- | ------------ | ------------------------------------------------------------- |
| [jsrt](http://db.jsrt.or.jp/eng.php) | abdomen           | segmentation | x_ray    | 2                    | 199/-/48             | 2021         | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

| class_name | Percentage of pixels（%） | Number of pictures in this category |
| ---------- | ------------------------- | ----------------------------------- |
| background | 6.46                      | 247                                 |
| heart      | 12.42                     | 247                                 |
| outer_zone | 50.66                     | 247                                 |
| lung       | 30.46                     | 247                                 |

### Visualization

![jsrt](https://github.com/uni-medical/medical-datasets-visualization/blob/main/2d/semantic_seg/x_ray/jsrt/jsrt_dataset.png?raw=true)

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- pillow(PIL)
- scikit-learn(sklearn)
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of PYTHONPATH, which should point to the project's directory so that Python can locate the module files. In jsrt/ root directory, run the following line to add the current directory to PYTHONPATH:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](http://db.jsrt.or.jp/eng.php) and decompression data to path 'data/jsrt'.
- run script `"python tools/prepare_dataset.py"` to split dataset and change folder structure as below.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── x_ray
  │   │   │   │   ├── jsrt
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

### Japanese Society of Radiological Technology Dataset (JSRT)

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                 config                                                                                 |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/x_ray/jsrt/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_bcss-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/x_ray/jsrt/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_bcss-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/x_ray/jsrt/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_bcss-512x512.py) |

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
