# RAVIR: A Dataset and Methodology for the Semantic Segmentation and Quantitative Analysis of Retinal Arteries and Veins in Infrared Reflectance Imaging

## Description

This project support **`RAVIR: A Dataset and Methodology for the Semantic Segmentation and Quantitative Analysis of Retinal Arteries and Veins in Infrared Reflectance Imaging`**, and the dataset used in this project can be downloaded from [here](https://ravir.grand-challenge.org/).

### Dataset Overview

The retinal vasculature provides important clues in the diagnosis and monitoring of systemic diseases including hypertension and diabetes. The microvascular system is of primary involvement in such conditions, and the retina is the only anatomical site where the microvasculature can be directly observed. The objective assessment of retinal vessels has long been considered a surrogate biomarker for systemic vascular diseases, and with recent advancements in retinal imaging and computer vision technologies, this topic has become the subject of renewed attention. In this paper, we present a novel dataset, dubbed RAVIR, for the semantic segmentation of Retinal Arteries and Veins in Infrared Reflectance (IR) imaging. It enables the creation of deep learning-based models that distinguish extracted vessel type without extensive post-processing.

### Original Statistic Information

| Dataset name                                | Anatomical region | Task type    | Modality                     | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                         |
| ------------------------------------------- | ----------------- | ------------ | ---------------------------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------- |
| [Ravir](https://ravir.grand-challenge.org/) | eye               | segmentation | infrared reflectance imaging | 3            | 23/-/19               | yes/-/-                | 2022         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |     23     |   87.22    |    -     |    -     |     -     |     -     |
|   artery   |     23     |    5.45    |    -     |    -     |     -     |     -     |
|    vein    |     23     |    7.33    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/infrared_reflectance_imaging/ravir/ravir_dataset.png)

## Dataset Citation

```bibtex
@article{hatamizadeh2022ravir,
  title={RAVIR: A dataset and methodology for the semantic segmentation and quantitative analysis of retinal arteries and veins in infrared reflectance imaging},
  author={Hatamizadeh, Ali and Hosseini, Hamid and Patel, Niraj and Choi, Jinseo and Pole, Cameron C and Hoeferlin, Cory M and Schwartz, Steven D and Terzopoulos, Demetri},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={26},
  number={7},
  pages={3272--3283},
  year={2022},
  publisher={IEEE}
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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `ravir/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://ravir.grand-challenge.org/) and decompression data to path `'data/ravir/'`.
- run script `"python tools/prepare_dataset.py"` to split dataset and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py --data_root data/ravir"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set can't be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── infrared_reflectance_imaging
  │   │   │   │   ├── ravir
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
  │   │   │   │   |   │   │   │   ├── yyy.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── yyy.png
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
| background |     18     |   87.41    |    5     |  86.53   |     -     |     -     |
|   artery   |     18     |    5.44    |    5     |   5.50   |     -     |     -     |
|    vein    |     18     |    7.15    |    5     |   7.97   |     -     |     -     |

### Training commands

To train models on a single server with one GPU. (default）

```shell
mim train mmseg ./configs/${CONFIG_PATH}
```

### Testing commands

To train models on a single server with one GPU. (default）

```shell
mim test mmseg ./configs/${CONFIG_PATH}  --checkpoint ${CHECKPOINT_PATH}
```

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/configs/fcn#results-and-models)

You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

## Results

### Ravir

|     Method      | Backbone | Crop Size |   lr   |                                   config                                   |
| :-------------: | :------: | :-------: | :----: | :------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  |  [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_ravir-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_ravir-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_ravir-512x512.py) |

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
