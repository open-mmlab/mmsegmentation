# Retinal Fundus Glaucoma  Challenge  Edition2

## Description

This project support **`Retinal Fundus Glaucoma  Challenge  Edition2`**, and the dataset used in this project can be downloaded from [here](https://refuge.grand-challenge.org/REFUGE2Download/).

### Dataset Overview

This regular-challenge dataset was provided by Sun Yat-sen Ophthalmic Center, Sun Yat-sen University, Guangzhou, China. The dataset contains 200 fundus color images: 100 pairs in the training set and 100 pairs in the test set.

### Original Statistic Information

| Dataset name                                                   | Anatomical region | Task type    | Modality        | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                         |
| -------------------------------------------------------------- | ----------------- | ------------ | --------------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------- |
| [Refuge2](https://refuge.grand-challenge.org/REFUGE2Download/) | eye               | segmentation | fundus photophy | 3            | 1200/400/-            | yes/-/-                | 2020         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    1200    |   98.34    |    -     |    -     |     -     |     -     |
| optic disc |    1200    |    1.22    |    -     |    -     |     -     |     -     |
| optic cup  |    1200    |    0.44    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/fundus_photography/refuge2/refuge2_dataset.png)

## Dataset Citation

```bibtex
@article{orlando2020refuge,
  title={Refuge challenge: A unified framework for evaluating automated methods for glaucoma assessment from fundus photographs},
  author={Orlando, Jos{\'e} Ignacio and Fu, Huazhu and Breda, Jo{\~a}o Barbosa and Van Keer, Karel and Bathula, Deepti R and Diaz-Pinto, Andr{\'e}s and Fang, Ruogu and Heng, Pheng-Ann and Kim, Jeyoung and Lee, JoonHo and others},
  journal={Medical image analysis},
  volume={59},
  pages={101570},
  year={2020},
  publisher={Elsevier}
}

@article{li2022development,
  title={Development and clinical deployment of a smartphone-based visual field deep learning system for glaucoma detection (vol 3, 123, 2020)},
  author={Li, Fei and Song, Diping and Chen, Han and Xiong, Jian and Li, Xingyi and Zhong, Hua and Tang, Guangxian and Fan, Sujie and Lam, Dennis SC and Pan, Weihua and others},
  journal={npj Digital Medicine},
  volume={5},
  number={1},
  year={2022},
  publisher={Nature Research (part of Springer Nature)}
}
No definitions available.
```

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- pillow(PIL) 9.3.0
- scikit-learn(sklearn) 1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `refuge2/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://refuge.grand-challenge.org/REFUGE2Download/) and decompression data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to split dataset and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set can't be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── fundus_photography
  │   │   │   │   ├── refuge2
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
  │   │   │   │   │   │   │   ├── val
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
| background |    960     |   98.35    |   240    |  98.33   |     -     |     -     |
| optic disc |    960     |    1.22    |   240    |   1.23   |     -     |     -     |
| optic cup  |    960     |    0.43    |   240    |   0.44   |     -     |     -     |

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

### Refuge2

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                          config                                                                                           |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/refuge2/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_refuge2-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/refuge2/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_refuge2-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/refuge2/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_refuge2-512x512.py) |

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
