# Colorectal Nuclear Segmentation and Phenotypes (CoNSeP) Dataset

## Description

This project supports **`Colorectal Nuclear Segmentation and Phenotypes (CoNSeP) Dataset`**, which can be downloaded from [here](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/).

### Dataset Overview

The CoNSeP (Colon Segmentation and Phenotyping) dataset consists of 41 H&E stained image tiles, each with a size of 1,000×1,000 pixels and a magnification of 40x. These images were extracted from 16 colorectal adenocarcinoma (CRA) whole slide images (WSI), each of which belonged to a separate patient and was scanned using an Omnyx VL120 scanner at the Pathology Department of the University Hospitals Coventry and Warwickshire NHS Trust, UK. This dataset was first used in  paper named, "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images".

### Original Statistic Information

| Dataset name                                             | Anatomical region | Task type    | Modality       | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License |
| -------------------------------------------------------- | ----------------- | ------------ | -------------- | ------------ | --------------------- | ---------------------- | ------------ | ------- |
| [CoNIC202](https://conic-challenge.grand-challenge.org/) | abdomen           | segmentation | histopathology | 7            | 4981/-/-              | yes/-/-                | 2022         | -       |

|           Class Name            | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :-----------------------------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|           background            |     27     |   83.61    |    14    |   80.4   |     -     |     -     |
|              other              |     17     |    0.17    |    9     |   0.52   |     -     |     -     |
|          inflammatory           |     25     |    2.66    |    14    |   2.14   |     -     |     -     |
|       healthy epithelial        |     3      |    1.47    |    2     |   1.58   |     -     |     -     |
| dysplastic/malignant epithelial |     10     |    7.17    |    8     |   9.16   |     -     |     -     |
|           fibroblast            |     23     |    3.84    |    14    |   4.63   |     -     |     -     |
|             muscle              |     8      |    1.05    |    3     |   1.42   |     -     |     -     |
|           endothelial           |     7      |    0.02    |    4     |   0.15   |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/histopathology/consep/consep_dataset.png)

### Prerequisites

- Python v3.8
- PyTorch v1.10.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `conic2022_seg/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://opendatalab.com/CoNSeP) and decompress data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set can't be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── histopathology
  │   │   │   │   ├── consep
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

## Dataset Citation

If this work is helpful for your research, please consider citing the below paper.

```
@article{graham2019hover,
  title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
  author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak, Jin Tae and Rajpoot, Nasir},
  journal={Medical Image Analysis},
  volume={58},
  pages={101563},
  year={2019},
  publisher={Elsevier}
}
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
