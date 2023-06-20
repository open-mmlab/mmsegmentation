# breastCancerCellSegmentation

## Description

This project supports **`breastCancerCellSegmentation`**, which can be downloaded from [here](https://www.heywhale.com/mw/dataset/5e9e9b35ebb37f002c625423).

### Dataset Overview

This dataset, with 58 H&E-stained histopathology images was used for breast cancer cell detection and associated real-world data.
Conventional histology uses a combination of hematoxylin and eosin stains, commonly referred to as H&E. These images are stained because most cells are inherently transparent with little or no intrinsic pigment.
Certain special stains selectively bind to specific components and can be used to identify biological structures such as cells.

### Original Statistic Information

| Dataset name                                                                                 | Anatomical region | Task type    | Modality       | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                         |
| -------------------------------------------------------------------------------------------- | ----------------- | ------------ | -------------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------- |
| [breastCancerCellSegmentation](https://www.heywhale.com/mw/dataset/5e9e9b35ebb37f002c625423) | cell              | segmentation | histopathology | 2            | 58/-/-                | yes/-/-                | 2020         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

|    Class Name    | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|    background    |     58     |   98.37    |    -     |    -     |     -     |     -     |
| breastCancerCell |     58     |    1.63    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/histopathology/breastCancerCellSegmentation/breastCancerCellSegmentation_dataset.png)

## Usage

### Prerequisites

- Python v3.8
- PyTorch v1.10.0
- pillow (PIL) v9.3.0
- scikit-learn (sklearn) v1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `breastCancerCellSegmentation/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- Download dataset from [here](https://www.heywhale.com/mw/dataset/5e9e9b35ebb37f002c625423) and save it to the `data/` directory .
- Decompress data to path `data/`. This will create a new folder named `data/breastCancerCellSegmentation/`, which contains the original image data.
- run script `python tools/prepare_dataset.py` to format data and change folder structure as below.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── histopathology
  │   │   │   │   ├── breastCancerCellSegmentation
  │   │   │   │   │   ├── configs
  │   │   │   │   │   ├── datasets
  │   │   │   │   │   ├── tools
  │   │   │   │   │   ├── data
  │   │   │   │   │   │   ├── breastCancerCellSegmentation
  |   │   │   │   │   │   │   ├── train.txt
  |   │   │   │   │   │   │   ├── val.txt
  |   │   │   │   │   │   │   ├── images
  |   │   │   │   │   │   │   |   ├── xxx.tif
  |   │   │   │   │   │   │   ├── masks
  |   │   │   │   │   │   │   |   ├── xxx.TIF

```

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
