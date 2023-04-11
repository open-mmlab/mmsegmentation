# SegPC2021

## Description

This project supports **`SegPC2021`**, which can be downloaded from [here](https://segpc-2021.grand-challenge.org/SegPC-2021/).

### Dataset Overview

Microscopic images were captured from bone marrow aspirate slides of patients diagnosed with Multiple Myeloma (MM), a type of white blood cancer. Slides were stained using Jenner-Giemsa stain, and plasma cells, which are cells of interest, must be segmented. Images were captured in raw BMP format using two cameras-

1. with a size of 2040x1536 pixels using cellSens software Version 2.1 (Olympus) attached to the microscope and
2. at a size of 1920x2560pixels from a Nikon camera attached to the microscope.

### Original Statistic Information

| Dataset name                                                    | Anatomical region | Task type    | Modality       | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                         |
| --------------------------------------------------------------- | ----------------- | ------------ | -------------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------- |
| [SegPC2021](https://segpc-2021.grand-challenge.org/SegPC-2021/) | bacteria          | segmentation | histopathology | 3            | 298/199/277           | yes/yes/no             | 2021         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    298     |   86.83    |   199    |  87.70   |     -     |     -     |
| cytoplasm  |    298     |    7.60    |   199    |   7.16   |     -     |     -     |
|  nucleus   |    298     |    5.57    |   199    |   5.14   |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/histopathology/segpc2021/segpc2021_dataset.png)

## Usage

### Prerequisites

- Python v3.8
- PyTorch v1.10.0
- pillow (PIL) v9.3.0
- scikit-learn (sklearn) v1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `segpc2021/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- Download dataset from [here](https://segpc-2021.grand-challenge.org/SegPC-2021/) and save it to the `data/` directory .
- Decompress data to path `data/`. This will create a new folder named `data/SegPC2021/`, which contains the original image data.
- run script `python tools/prepare_dataset.py` to format data and change folder structure as below.
- run script `python ../../tools/split_seg_dataset.py` to split dataset. For the Bacteria_detection dataset, as there is no test or validation dataset, we sample 20% samples from the whole dataset as the validation dataset and 80% samples for training data and make two filename lists `train.txt` and `val.txt`. As we set the random seed as the hard code, we eliminated the randomness, the dataset split actually can be reproducible.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── histopathology
  │   │   │   │   ├── segpc2021
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
  │   │   │   │   │   │   │   ├── test
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png
  │   │   │   │   │   │   │   ├── val
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png
  │   │   │   │   │   │   ├── masks
  │   │   │   │   │   │   │   ├── train
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png
  │   │   │   │   │   │   │   ├── val
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png
```

### Divided Dataset Information

***Note: The table information below is divided by ourselves.***

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    298     |   86.83    |   199    |  87.70   |     -     |     -     |
| cytoplasm  |    298     |    7.60    |   199    |   7.16   |     -     |     -     |
|  nucleus   |    298     |    5.57    |   199    |   5.14   |     -     |     -     |

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

### SegPC2021

***Note: The following experimental results are based on the data randomly partitioned according to the above method described in the dataset preparing section.***

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                     config                                     |         download         |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :----------------------------------------------------------------------------: | :----------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_segpc2021-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_segpc2021-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_segpc2021-512x512.py) | [model](<>) \| [log](<>) |

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
