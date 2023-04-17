# NuCLS

## Description

This project supports **`NuCLS`**, which can be downloaded from [here](https://sites.google.com/view/nucls/home).

### Dataset Overview

The NuCLS dataset contains over 220,000 labeled nuclei from breast cancer images from TCGA. These nuclei were annotated through the collaborative effort of pathologists, pathology residents, and medical students using the Digital Slide Archive. These data can be used in several ways to develop and validate algorithms for nuclear detection, classification, and segmentation, or as a resource to develop and evaluate methods for interrater analysis.
Data from both single-rater and multi-rater studies are provided. For single-rater data we provide both pathologist-reviewed and uncorrected annotations. For multi-rater datasets we provide annotations generated with and without suggestions from weak segmentation and classification algorithms.
For more details consult our GigaScience paper, or contact us directly with questions.

### Original Statistic Information

| Dataset name                                      | Anatomical region | Task type    | Modality       | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                         |
| ------------------------------------------------- | ----------------- | ------------ | -------------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------- |
| [NuCLS](https://sites.google.com/view/nucls/home) | cell              | segmentation | histopathology | 3            | 1337/-/-              | yes/-/-                | 2017         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

|      Class Name      | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :------------------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|      background      |    1337    |   73.03    |    -     |    -     |     -     |     -     |
|      lymphocyte      |    877     |    3.83    |    -     |    -     |     -     |     -     |
|      fibroblast      |    909     |    5.48    |    -     |    -     |     -     |     -     |
|     plasma_cell      |    382     |    0.83    |    -     |    -     |     -     |     -     |
|        tumor         |    1074    |   14.42    |    -     |    -     |     -     |     -     |
|      macrophage      |     65     |    0.15    |    -     |    -     |     -     |     -     |
| vascular_endothelium |    105     |    0.39    |    -     |    -     |     -     |     -     |
|    myoepithelium     |     1      |    0.00    |    -     |    -     |     -     |     -     |
|    mitotic_figure    |     20     |    0.03    |    -     |    -     |     -     |     -     |
|      neutrophil      |     8      |    0.01    |    -     |    -     |     -     |     -     |
|    apoptotic_body    |     27     |    0.02    |    -     |    -     |     -     |     -     |
|      unlabeled       |    601     |    1.81    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/histopathology/NuCLS/NuCLS_dataset.png)

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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `NuCLS/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- Download dataset from [here](https://sites.google.com/view/nucls/home) and save it to the `data/` directory .
- Decompress data to path `data/`. This will create a new folder named `data/NuCLS/`, which contains the original image data.
- run script `python tools/prepare_dataset.py` to format data and change folder structure as below.
- run script `python ../../tools/split_seg_dataset.py` to split dataset. For the Bacteria_detection dataset, as there is no test or validation dataset, we sample 20% samples from the whole dataset as the validation dataset and 80% samples for training data and make two filename lists `train.txt` and `val.txt`. As we set the random seed as the hard code, we eliminated the randomness, the dataset split actually can be reproducible.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── histopathology
  │   │   │   │   ├── NuCLS
  │   │   │   │   │   ├── configs
  │   │   │   │   │   ├── datasets
  │   │   │   │   │   ├── tools
  │   │   │   │   │   ├── data
  │   │   │   │   │   │   ├── train.txt
  │   │   │   │   │   │   ├── val.txt
  │   │   │   │   │   │   ├── Bacteria_detection_with_darkfield_microscopy_datasets
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

|      Class Name      | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :------------------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|      background      |    1069    |   73.07    |    -     |    -     |     -     |     -     |
|      lymphocyte      |    692     |    3.79    |    -     |    -     |     -     |     -     |
|      fibroblast      |    730     |    5.47    |    -     |    -     |     -     |     -     |
|     plasma_cell      |    295     |    0.76    |    -     |    -     |     -     |     -     |
|        tumor         |    858     |   14.44    |    -     |    -     |     -     |     -     |
|      macrophage      |     56     |    0.16    |    -     |    -     |     -     |     -     |
| vascular_endothelium |     84     |    0.40    |    -     |    -     |     -     |     -     |
|    myoepithelium     |     1      |    0.00    |    -     |    -     |     -     |     -     |
|    mitotic_figure    |     19     |    0.03    |    -     |    -     |     -     |     -     |
|      neutrophil      |     8      |    0.01    |    -     |    -     |     -     |     -     |
|    apoptotic_body    |     22     |    0.02    |    -     |    -     |     -     |     -     |
|      unlabeled       |    485     |    1.85    |    -     |    -     |     -     |     -     |

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

### NuCLS

***Note: The following experimental results are based on the data randomly partitioned according to the above method described in the dataset preparing section.***

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                   config                                   |         download         |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :------------------------------------------------------------------------: | :----------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_NuCLS-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_NuCLS-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_NuCLS-512x512.py) | [model](<>) \| [log](<>) |

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
