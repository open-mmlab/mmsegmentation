# Vampire with darkfield microscopy Dataset

This project support **`Vampire with darkfield microscopy Dataset`**, and the dataset can be download from [here](https://tianchi.aliyun.com/dataset/94411).

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc3
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.1.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc3

### Dataset preparing

- download dataset from [here](https://tianchi.aliyun.com/dataset/94411) and decompression data to path 'data/vampire'.
- run script `"python projects/vampire/tools/prepare_dataset.py"` to split dataset and change folder structure as below.

```none
  mmsegmentation
  ├── mmseg
  ├── tools
  ├── configs
  ├── data
  │   ├── vampire
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

### Training commands

```shell
mim train mmseg projects/vampire/configs/${CONFIG_PATH}
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```shell
mim train mmseg projects/vampire/configs/${CONFIG_PATH}  --launcher pytorch --gpus 8
```

### Testing commands

```shell
mim test mmseg projects/vampire/configs/${CONFIG_PATH}  --checkpoint ${CHECKPOINT_PATH}
```

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/configs/fcn#results-and-models)

You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

## Results

### Bactteria detection with darkfield microscopy Dataset

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                   config                                                                   |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :----------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 91.68 | 95.55 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/vampire/configs/Bactteria_Det_unet_0.01_CrossEntropyLoss.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 92.02 | 95.74 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/vampire/configs/Bactteria_Det_unet_0.001_CrossEntropyLoss.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 90.25 | 94.72 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/vampire/configs/Bactteria_Det_unet_0.0001_CrossEntropyLoss.py) |

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
