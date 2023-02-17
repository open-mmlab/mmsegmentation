# Bactteria detection with darkfield microscopy

## Description

This project support **`Bactteria detection with darkfield microscopy `**, and the dataset used in this project can be downloaded from [here](https://tianchi.aliyun.com/dataset/94411).

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- pillow(PIL)
- scikit-learn(sklearn)
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

### Dataset preparing

- download dataset from [here](https://tianchi.aliyun.com/dataset/94411) and decompression data to path 'data/bactteria_detection'.
- run script `"python projects/bactteria_detection/tools/prepare_dataset.py"` to split dataset and change folder structure as below.

```none
  mmsegmentation
  ├── mmseg
  ├── tools
  ├── configs
  ├── data
  │   ├── bactteria_detection
  │   │   ├── images
  │   │   │   ├── train
  |   │   │   │   ├── xxx.png
  |   │   │   │   ├── ...
  |   │   │   │   └── xxx.png
  │   │   │   ├── val
  |   │   │   │   ├── yyy.png
  |   │   │   │   ├── ...
  |   │   │   │   └── yyy.png
  │   │   ├── masks
  │   │   │   ├── train
  |   │   │   │   ├── xxx.png
  |   │   │   │   ├── ...
  |   │   │   │   └── xxx.png
  │   │   │   ├── val
  |   │   │   │   ├── yyy.png
  |   │   │   │   ├── ...
  |   │   │   │   └── yyy.png
```

All the commands below rely on the correct configuration of PYTHONPATH, which should point to the project's directory so that Python can locate the module files. In bactteria_detection/ root directory, run the following line to add the current directory to PYTHONPATH:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Training commands

```shell
mim train mmseg projects/bactteria_detection/configs/${CONFIG_PATH}
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```shell
mim train mmseg projects/bactteria_detection/configs/${CONFIG_PATH}  --launcher pytorch --gpus 8
```

### Testing commands

```shell
mim test mmseg projects/bactteria_detection/configs/${CONFIG_PATH}  --checkpoint ${CHECKPOINT_PATH}
```

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/configs/fcn#results-and-models)

You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

## Results

### Bactteria detection with darkfield microscopy

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                    config                                                                                     |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/bactteria_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_bactteria-detection-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/bactteria_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_bactteria-detection-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/bactteria_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_bactteria-detection-512x512.py) |

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
