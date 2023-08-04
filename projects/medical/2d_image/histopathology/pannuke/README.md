# Pan-Cancer Histology Dataset for Nuclei Instance Segmentation and Classification (PanNuke)

## Description

This project supports **`Pan-Cancer Histology Dataset for Nuclei Instance Segmentation and Classification (PanNuke)`**, which can be downloaded from [here](https://academictorrents.com/details/99f2c7b57b95500711e33f2ee4d14c9fd7c7366c).

### Dataset Overview

Semi automatically generated nuclei instance segmentation and classification dataset with exhaustive nuclei labels across 19 different tissue types. The dataset consists of 481 visual fields, of which 312 are randomly sampled from more than 20K whole slide images at different magnifications, from multiple data sources. In total the dataset contains 205,343 labeled nuclei, each with an instance segmentation mask. Models trained on pannuke can aid in whole slide image tissue type segmentation, and generalise to new tissues. PanNuke demonstrates one of the first successfully semi-automatically generated datasets.

### Statistic Information

| Dataset Name                                                                             | Anatomical Region | Task Type    | Modality       | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                         |
| ---------------------------------------------------------------------------------------- | ----------------- | ------------ | -------------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------- |
| [Pannuke](https://academictorrents.com/details/99f2c7b57b95500711e33f2ee4d14c9fd7c7366c) | full_body         | segmentation | histopathology | 6            | 7901/-/-              | yes/-/-                | 2019         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

|        Class Name         | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :-----------------------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|        background         |    7901    |   83.32    |    -     |    -     |     -     |     -     |
|        neoplastic         |    4190    |    8.64    |    -     |    -     |     -     |     -     |
| non-neoplastic epithelial |    4126    |    1.77    |    -     |    -     |     -     |     -     |
|       inflammatory        |    6137    |    3.73    |    -     |    -     |     -     |     -     |
|        connective         |    232     |    0.07    |    -     |    -     |     -     |     -     |
|           dead            |    1528    |    2.47    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![pannuke](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/histopathology/pannuke/pannuke_dataset.png?raw=true)

### Dataset Citation

```
@inproceedings{gamper2019pannuke,
  title={PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification},
  author={Gamper, Jevgenij and Koohbanani, Navid Alemi and Benet, Ksenija and Khuram, Ali and Rajpoot, Nasir},
  booktitle={European Congress on Digital Pathology},
  pages={11--19},
  year={2019},
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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `pannuke/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- download dataset from [here](https://academictorrents.com/details/99f2c7b57b95500711e33f2ee4d14c9fd7c7366c) and decompress data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set cannot be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── histopathology
  │   │   │   │   ├── pannuke
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

|        Class Name         | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :-----------------------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|        background         |    6320    |   83.38    |   1581   |   83.1   |     -     |     -     |
|        neoplastic         |    3339    |    8.55    |   851    |   9.0    |     -     |     -     |
| non-neoplastic epithelial |    3293    |    1.77    |   833    |   1.76   |     -     |     -     |
|       inflammatory        |    4914    |    3.72    |   1223   |   3.76   |     -     |     -     |
|        connective         |    170     |    0.06    |    62    |   0.09   |     -     |     -     |
|           dead            |    1235    |    2.51    |   293    |   2.29   |     -     |     -     |

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
