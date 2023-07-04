# Retinal Images vessel Tree Extraction (RITE)

## Description

This project supports **`Retinal Images vessel Tree Extraction (RITE) `**, which can be downloaded from [here](https://opendatalab.com/RITE).

### Dataset Overview

The RITE (Retinal Images vessel Tree Extraction) is a database that enables comparative studies on segmentation or classification of arteries and veins on retinal fundus images, which is established based on the public available DRIVE database (Digital Retinal Images for Vessel Extraction). RITE contains 40 sets of images, equally separated into a training subset and a test subset, the same as DRIVE. The two subsets are built from the corresponding two subsets in DRIVE. For each set, there is a fundus photograph, a vessel reference standard. The fundus photograph is inherited from DRIVE. For the training set, the vessel reference standard is a modified version of 1st_manual from DRIVE. For the test set, the vessel reference standard is 2nd_manual from DRIVE.

### Statistic Information

| Dataset Name                         | Anatomical Region | Task Type    | Modality           | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                         |
| ------------------------------------ | ----------------- | ------------ | ------------------ | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------- |
| [Rite](https://opendatalab.com/RITE) | head_and_neck     | segmentation | fundus_photography | 2            | 20/-/20               | yes/-/yes              | 2013         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |     20     |   91.61    |    -     |    -     |    20     |   91.58   |
|   vessel   |     20     |    8.39    |    -     |    -     |    20     |   8.42    |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![rite](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/fundus_photography/rite/rite_dataset.png?raw=true)

### Dataset Citation

```
@InProceedings{10.1007/978-3-642-40763-5_54,
	author={Hu, Qiao and Abr{\`a}moff, Michael D. and Garvin, Mona K.},
	title={Automated Separation of Binary Overlapping Trees in Low-Contrast Color Retinal Images},
	booktitle={Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2013},
	year={2013},
	pages={436--443},
}


```

### Prerequisites

- Python v3.8
- PyTorch v1.10.0
- pillow(PIL) v9.3.0 9.3.0
- scikit-learn(sklearn) v1.2.0 1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `rite/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- download dataset from [here](https://opendatalab.com/RITE) and decompress data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set cannot be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

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

### Training commands

To train models on a single server with one GPU. (default)

```shell
mim train mmseg ./configs/${CONFIG_FILE}
```

### Testing commands

To test models on a single server with one GPU. (default)

```shell
mim test mmseg ./configs/${CONFIG_FILE}  --checkpoint ${CHECKPOINT_PATH}
```

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/configs/fcn#results-and-models)

You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code
  - [x] Basic docstrings & proper citation
  - [ ] Test-time correctness
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
