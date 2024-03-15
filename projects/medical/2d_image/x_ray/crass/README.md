# Chest Radiograph Anatomical Structure Segmentation (CRASS)

## Description

This project supports **`Chest Radiograph Anatomical Structure Segmentation (CRASS) `**, which can be downloaded from [here](https://crass.grand-challenge.org/).

### Dataset Overview

A set of consecutively obtained posterior-anterior chest radiograph were selected from a database containing images acquired at two sites in sub Saharan Africa with a high tuberculosis incidence. All subjects were 15 years or older. Images from digital chest radiography units were used (Delft Imaging Systems, The Netherlands) of varying resolutions, with a typical resolution of 1800--2000 pixels, the pixel size was 250 lm isotropic. From the total set of images, 225 were considered to be normal by an expert radiologist, while 333 of the images contained abnormalities. Of the abnormal images, 220 contained abnormalities in the upper area of the lung where the clavicle is located. The data was divided into a training and a test set. The training set consisted of 299 images, the test set of 249 images.
The current data is still incomplete and to be added later.

### Information Statistics

| Dataset Name                                | Anatomical Region | Task Type    | Modality | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                       |
| ------------------------------------------- | ----------------- | ------------ | -------- | ------------ | --------------------- | ---------------------- | ------------ | ------------------------------------------------------------- |
| [crass](https://crass.grand-challenge.org/) | pulmonary         | segmentation | x_ray    | 2            | 299/-/234             | yes/-/no               | 2021         | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    299     |   98.38    |    -     |    -     |     -     |     -     |
| clavicles  |    299     |    1.62    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![crass](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/x_ray/crass/crass_dataset.png?raw=true)

### Dataset Citation

```
@article{HOGEWEG20121490,
	title={Clavicle segmentation in chest radiographs},
	journal={Medical Image Analysis},
	volume={16},
	number={8},
	pages={1490-1502},
	year={2012}
}
```

### Prerequisites

- Python v3.8
- PyTorch v1.10.0
- pillow(PIL) v9.3.0
- scikit-learn(sklearn) v1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `crass/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- download dataset from [here](https://crass.grand-challenge.org/) and decompress data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set cannot be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── x_ray
  │   │   │   │   ├── crass
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

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    227     |   98.38    |    57    |  98.39   |     -     |     -     |
| clavicles  |    227     |    1.62    |    57    |   1.61   |     -     |     -     |

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

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings
  - [ ] Unit tests
  - [ ] Code polishing
  - [ ] Metafile.yml

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
