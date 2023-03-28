# Endoscopic Vision Challenge 2015 (EndoVis15)

## Description

This project supports **`Endoscopic Vision Challenge 2015 (EndoVis15) `**, which can be downloaded from [here](https://polyp.grand-challenge.org/Databases/).

### Dataset Overview

EndoVis15 is a database of frames extracted from colonoscopy videos. These frames contain several examples of polyps. This ground truth consists of a mask corresponding to the region covered by the polyp in the image.

### Information Statistics

| Dataset Name                                              | Anatomical Region | Task Type    | Modality  | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                   |
| --------------------------------------------------------- | ----------------- | ------------ | --------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------- |
| [Endovis15](https://polyp.grand-challenge.org/Databases/) | pelvis            | segmentation | endoscopy | 2            | 612/-/-               | yes/-/-                | 2017         | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    612     |   90.83    |    -     |    -     |     -     |     -     |
|   polyp    |    612     |    9.17    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![endovis15](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/endoscopy_images/endovis15/endovis15_dataset.png?raw=true)

### Dataset Citation

```
@article{c058b44d5ac04a4f8fda96db6b3260ff,
	title={Comparative Validation of Polyp Detection Methods in Video Colonoscopy: Results from the MICCAI 2015 Endoscopic Vision Challenge},
	author={Jorge Bernal and Nima Tajkbaksh and Sanchez, {Francisco Javier} and Matuszewski, {Bogdan J.} and Hao Chen and Lequan Yu and Quentin Angermann and Olivier Romain and Bjorn Rustad and Ilangko Balasingham and Konstantin Pogorelov and Sungbin Choi and Quentin Debard and Lena Maier-Hein and Stefanie Speidel and Danail Stoyanov and Patrick Brandao and Henry Cordova and Cristina Sanchez-Montes and Gurudu, {Suryakanth R.} and Gloria Fernandez-Esparrach and Xavier Dray and Jianming Liang and Aymeric Histace},
	journal={IEEE Transactions on Medical Imaging},
	volume={36},
	number={6},
	pages={1231--1249},
	year={2017}
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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `endovis15/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- download dataset from [here](https://polyp.grand-challenge.org/Databases/) and decompress data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set cannot be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── endoscopy
  │   │   │   │   ├── endovis15
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
| background |    489     |   91.28    |   123    |  89.07   |     -     |     -     |
|   polyp    |    489     |    8.72    |   123    |  10.93   |     -     |     -     |

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

## Results

### Endoscopic Vision Challenge 2015 (EndoVis15)

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                        config                                                                                        |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/endoscopy/endovis15/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_endovis15-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/endoscopy/endovis15//configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_endovis15-512x512.py) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/endoscopy/endovis15/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_endovis15-512x512.py) |

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
