# Bacteria detection with darkfield microscopy Dataset

This project support **`Bacteria detection with darkfield microscopy Dataset`**, and the dataset can be download from [here](https://tianchi.aliyun.com/dataset/94411).

## Dataset preparing

Preparing `Bacteria detection with darkfield microscopy Dataset` dataset in following format as below.

```none
  mmsegmentation
  ├── mmseg
  ├── tools
  ├── configs
  ├── data
  │   ├── Bacteria_Det
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

- download dataset from [here](https://tianchi.aliyun.com/dataset/94411) and decompression data to path 'data/Bacteria_Det'.
- run script `"python projects/Bacteria_Dataset/tools/random_split.py"` to split dataset and change folder structure.

## Training commands with config in `configs`

```bash
# Dataset train commands example
# at `mmsegmentation` folder
bash tools/dist_train.sh projects/Bacteria_Dataset/configs/Bactteria_Det_unet_0.01_CrossEntropyLoss.py 4
```

## Results

### Bacteria detection with darkfield microscopy Dataset

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                       config                                                                        |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :-------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 91.68 | 95.55 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/Bacteria_Dataset/configs/Bactteria_Det_unet_0.01_CrossEntropyLoss.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 92.02 | 95.74 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/Bacteria_Dataset/configs/Bactteria_Det_unet_0.001_CrossEntropyLoss.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 90.25 | 94.72 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/Bacteria_Dataset/configs/Bactteria_Det_unet_0.0001_CrossEntropyLoss.py) |

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
