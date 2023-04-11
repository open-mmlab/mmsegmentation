# QUBIQ2020

## Description

This project supports **`QUBIQ2020`**, which can be downloaded from [here](https://syncandshare.lrz.de/public?folderID=MlJqWDRqTk1yOVBDUzd6UTNvdkhm).

### Dataset Overview

Training and test data will comprise 7 binary segmentation tasks in four different CT and MR data sets. Some of the data sets contain more than one binary segmentation (sub-)task, e.g., different sub-structures of tumor or anatomy need to be segmented.

All data sets have about 50 to 100 cases featuring one selected 2D slice each. Each structure of interest is segmented between three and seven times by different experts, and individual segmentations are made available. The task is to delineate structures in a given slice, and to match the distribution – or spread – of the expert's annotations well.

The data is split accordingly in four different image sets, and for each case in those sets, binary labels are given for each segmentation task.

The following data and tasks are available:

- Prostate images (MRI): 55 cases, two segmentation tasks, six annotations (except one subject has only 5 annotations);
- Brain growth images (MRI): 39 cases, one segmentation task, seven annotations;
- Brain tumor images (multimodal MRI): 32 cases, three segmentations tasks, three annotations \[Please note: this data set will receive the additional case in the near future\];
- Kidney images (CT): 24 cases, one segmentation task, three annotations;

Data is available as .nii files with 2D slices. For “Brain tumor” the file contains slices of all four MR modalities.

### Original Statistic Information

| Dataset name                                                                          | Anatomical region | Task type    | Modality | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                         |
| ------------------------------------------------------------------------------------- | ----------------- | ------------ | -------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------- |
| [QUBIQ2020](https://syncandshare.lrz.de/public?folderID=MlJqWDRqTk1yOVBDUzd6UTNvdkhm) | tumor             | segmentation | CT       | 3            | 162/18/-              | yes/yes/-              | 2020         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| Class Name  | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :---------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background  |    162     |   95.76    |    18    |  96.29   |     -     |     -     |
| brainGrowth |    102     |    2.94    |    10    |   2.22   |     -     |     -     |
|   kidney    |     60     |    1.30    |    8     |   1.49   |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/ct/QUBIQ2020/QUBIQ2020_dataset.png)

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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `QUBIQ2020/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- Download dataset from [here](https://syncandshare.lrz.de/public?folderID=MlJqWDRqTk1yOVBDUzd6UTNvdkhm) and save it to the `data/` directory .
- Decompress data to path `data/`. This will create a new folder named `data/QUBIQ2020/`, which contains the original image data.
- run script `python tools/prepare_dataset.py` to format data and change folder structure as below.
- run script `python ../../tools/split_seg_dataset.py` to split dataset. For the QUBIQ2020 dataset, as there is no test or validation dataset, we sample 20% samples from the whole dataset as the validation dataset and 80% samples for training data and make two filename lists `train.txt` and `val.txt`. As we set the random seed as the hard code, we eliminated the randomness, the dataset split actually can be reproducible.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── ct
  │   │   │   │   ├── QUBIQ2020
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

| Class Name  | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :---------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background  |    162     |   95.76    |    18    |  96.29   |     -     |     -     |
| brainGrowth |    102     |    2.94    |    10    |   2.22   |     -     |     -     |
|   kidney    |     60     |    1.30    |    8     |   1.49   |     -     |     -     |

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

### QUBIQ2020

***Note: The following experimental results are based on the data randomly partitioned according to the above method described in the dataset preparing section.***

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                     config                                     |         download         |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :----------------------------------------------------------------------------: | :----------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_QUBIQ2020-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_QUBIQ2020-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_QUBIQ2020-512x512.py) | [model](<>) \| [log](<>) |

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
