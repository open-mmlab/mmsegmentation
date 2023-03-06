# RIM-ONE DL: A unified retinal image database for assessing glaucoma using Deep Learning

## Description

This project support **`RIM-ONE DL: A unified retinal image database for assessing glaucoma using Deep Learning`**, and the dataset used in this project can be downloaded from [here](https://bit.ly/rim-one-dl-images).

### Dataset Overview

The RIM-ONE DL image dataset consists of 313 retinographies from normal subjects and 172 retinographies from patients with glaucoma. These images were captured in three Spanish hospitals: Hospital Universitario de Canarias (HUC), in Tenerife, Hospital Universitario Miguel Servet (HUMS), in Zaragoza, and Hospital Clínico Universitario San Carlos (HCSC), in Madrid.

### Original Statistic Information

| Dataset name                                   | Anatomical region | Task type    | Modality        | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                         |
| ---------------------------------------------- | ----------------- | ------------ | --------------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------- |
| [RIM_ONE_DL](https://bit.ly/rim-one-dl-images) | eye               | segmentation | fundus photophy | 3            | 339/-/146             | yes/-/yes              | 2020         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    339     |   59.54    |    -     |    -     |    146    |   57.78   |
| optic disc |    339     |   29.71    |    -     |    -     |    146    |   31.15   |
| optic cup  |    339     |   10.75    |    -     |    -     |    146    |   11.07   |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/fundus_photography/rimonedl/rimonedl_dataset.png)

## Dataset Citation
```bibtex
@article{RIMONEDLImageAnalStereol2346,
	author = {Francisco José Fumero Batista and Tinguaro Diaz-Aleman and Jose Sigut and Silvia Alayon and Rafael Arnay and Denisse Angel-Pereira},
	title = {RIM-ONE DL: A Unified Retinal Image Database for Assessing Glaucoma Using Deep Learning},
	journal = {Image Analysis & Stereology},
	volume = {39},
	number = {3},
	year = {2020},
	keywords = {Convolutional Neural Networks; Deep Learning; Glaucoma Assessment; RIM-ONE},
	issn = {1854-5165},
	pages = {161--167},
	doi = {10.5566/ias.2346},
	url = {https://www.ias-iss.org/ojs/IAS/article/view/2346}
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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `rimonedl/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://bit.ly/rim-one-dl-images) and decompression data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to split dataset and change folder structure as below.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── fundus_photography
  │   │   │   │   ├── rimonedl
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
  │   │   │   │   │   │   │   ├── test
  │   │   │   │   |   │   │   │   ├── yyy.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── yyy.png
  │   │   │   │   │   │   ├── masks
  │   │   │   │   │   │   │   ├── train
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png
  │   │   │   │   │   │   │   ├── test
  │   │   │   │   |   │   │   │   ├── yyy.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── yyy.png
```

### Training commands

To train models on a single server with one GPU. (default）

```shell
mim train mmseg ./configs/${CONFIG_PATH}
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```shell
mim train mmseg ./configs/${CONFIG_PATH}  --launcher pytorch --gpus 8
```

### Testing commands

To train models on a single server with one GPU. (default）

```shell
mim test mmseg ./configs/${CONFIG_PATH}  --checkpoint ${CHECKPOINT_PATH}
```

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/configs/fcn#results-and-models)

You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

## Results

### rimonedl

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                           config                                                                                            |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/rimonedl/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_rimonedl-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/rimonedl/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_rimonedl-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/rimonedl/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_rimonedl-512x512.py) |

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
