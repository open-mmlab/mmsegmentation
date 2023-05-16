# Brain CT Images with Intracranial Hemorrhage Masks (Cranium)

## Description

This project supports **`Brain CT Images with Intracranial Hemorrhage Masks (Cranium)`**, which can be downloaded from [here](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images).

### Dataset Overview

This dataset consists of head CT (Computed Thomography) images in jpg format. There are 2500 brain window images and 2500 bone window images, for 82 patients. There are approximately 30 image slices per patient. 318 images have associated intracranial image masks. Also included are csv files containing hemorrhage diagnosis data and patient data.
This is version 1.0.0 of this dataset. A full description of this dataset as well as updated versions can be found here:
https://physionet.org/content/ct-ich/1.0.0/

### Statistic Information

| Dataset Name                                                                        | Anatomical Region | Task Type    | Modality | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                   |
| ----------------------------------------------------------------------------------- | ----------------- | ------------ | -------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------- |
| [Cranium](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images) | head_and_neck     | segmentation | ct       | 2            | 2501/-/-              | yes/-/-                | 2020         | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    2501    |   99.93    |    -     |    -     |     -     |     -     |
| hemorrhage |    318     |    0.07    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![cranium](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/ct/cranium/cranium_dataset.png?raw=true)

## Dataset Citation

```
@article{hssayeni2020computed,
	title={Computed tomography images for intracranial hemorrhage detection and segmentation},
	author={Hssayeni, Murtadha and Croock, MS and Salman, AD and Al-khafaji, HF and Yahya, ZA and Ghoraani, B},
	journal={Intracranial Hemorrhage Segmentation Using A Deep Convolutional Model. Data},
	volume={5},
	number={1},
	pages={179},
	year={2020}
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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `cranium/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- download dataset from [here](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images) and decompress data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set cannot be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── ct
  │   │   │   │   ├── cranium
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
| background |    2000    |   99.93    |   501    |  99.92   |     -     |     -     |
| hemorrhage |    260     |    0.07    |   260    |   0.08   |     -     |     -     |

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
