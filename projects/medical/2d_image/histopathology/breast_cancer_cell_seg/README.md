# Breast Cancer Cell Segmentation

## Description

This project support **`Breast Cancer Cell Segmentation`**, and the dataset used in this project can be downloaded from [here](https://tianchi.aliyun.com/dataset/dataDetail?dataId=90152).

### Dataset Overview

In this dataset, there are 58 H&E stained histopathology images used in breast cancer cell detection with associated ground truth data available. Routine histology uses the stain combination of hematoxylin and eosin, commonly referred to as H&E. These images are stained since most cells are essentially transparent, with little or no intrinsic pigment. Certain special stains, which bind selectively to particular components, are be used to identify biological structures such as cells. In those images, the challenging problem is cell segmentation for subsequent classification in benign and malignant cells.

### Original Statistic Information

| Dataset name                                                                                  | Anatomical region | Task type    | Modality       | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                                                                |
| --------------------------------------------------------------------------------------------- | ----------------- | ------------ | -------------- | ------------ | --------------------- | ---------------------- | ------------ | ------------------------------------------------------------------------------------------------------ |
| [Breast Cancer Cell Segmentation](https://tianchi.aliyun.com/dataset/dataDetail?dataId=90152) | thorax            | segmentation | histopathology | 2            | 58/-/-                | yes/-/-                | 2021         | [CC-BY-SA-NC 4.0](http://creativecommons.org/licenses/by-sa/4.0/?spm=5176.12282016.0.0.3f5b5291ypBxb2) |

|     Class Name     | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :----------------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|       normal       |     58     |   98.37    |    -     |    -     |     -     |     -     |
| breast cancer cell |     58     |    1.63    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/histopathology/breast_cancer_cell_seg/breast_cancer_cell_seg_dataset.png)

## Dataset Citation

```
@inproceedings{gelasca2008evaluation,
  title={Evaluation and benchmark for biological image segmentation},
  author={Gelasca, Elisa Drelie and Byun, Jiyun and Obara, Boguslaw and Manjunath, BS},
  booktitle={2008 15th IEEE international conference on image processing},
  pages={1816--1819},
  year={2008},
  organization={IEEE}
}
```

### Prerequisites

- Python v3.8
- PyTorch v1.10.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `breast_cancer_cell_seg/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://tianchi.aliyun.com/dataset/dataDetail?dataId=90152) and decompression data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set can't be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── histopathology
  │   │   │   │   ├── breast_cancer_cell_seg
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
|  background  |     46     |   98.36    |    12    |  98.41   |     -     |     -     |
| erythrocytes |     46     |    1.64    |    12    |   1.59   |     -     |     -     |

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

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

  - [x] Basic docstrings & proper citation

  - [x] Test-time correctness

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
