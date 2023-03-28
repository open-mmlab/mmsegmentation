# Japanese Society of Radiological Technology Dataset (JSRT)

## Description

This project supports **`Japanese Society of Radiological Technology Dataset (JSRT) `**, which can be downloaded from [here](http://db.jsrt.or.jp/eng.php).

### Dataset Overview

The chest X-ray image is recorded in png file form, with 199 cases for training and 48 cases for test. The label image is multi-labeled image of the lung field area. The definition and determination of the lung field regions in the label data are not medically supervised, so there is no medical basis for them.

### Information Statistics

| Dataset Name                         | Anatomical Region | Task Type    | Modality | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                       |
| ------------------------------------ | ----------------- | ------------ | -------- | ------------ | --------------------- | ---------------------- | ------------ | ------------------------------------------------------------- |
| [jsrt](http://db.jsrt.or.jp/eng.php) | abdomen           | segmentation | x_ray    | 2            | 199/-/48              | yes/-/yes              | 2021         | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    247     |    6.46    |    -     |    -     |     -     |     -     |
|   heart    |    247     |   12.42    |    -     |    -     |     -     |     -     |
| outer_zone |    247     |   50.66    |    -     |    -     |     -     |     -     |
|    lung    |    247     |   30.46    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![jsrt](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/x_ray/jsrt/jsrt_dataset.png?raw=true)

### Dataset Citation

```
@article{doi:10.2214/ajr.174.1.1740071,
	author={Shiraishi, Junji and Katsuragawa, Shigehiko and Ikezoe, Junpei and Matsumoto, Tsuneo and Kobayashi, Takeshi and Komatsu, Ken-ichi and Matsui, Mitate and Fujita, Hiroshi and Kodera, Yoshie and Doi, Kunio},
	title={Development of a Digital Image Database for Chest Radiographs With and Without a Lung Nodule},
	journal={American Journal of Roentgenology},
	volume={174},
	number={1},
	pages={71--74},
	year={2000}
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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `jsrt/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- download dataset from [here](http://db.jsrt.or.jp/eng.php) and decompress data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set cannot be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── x_ray
  │   │   │   │   ├── jsrt
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
| background |    199     |    6.90    |    -     |    -     |    48     |   6.02    |
|   heart    |    199     |   12.49    |    -     |    -     |    48     |   12.36   |
| outer_zone |    199     |   49.73    |    -     |    -     |    48     |   51.59   |
|    lung    |    199     |   30.89    |    -     |    -     |    48     |   30.03   |

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

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/configs/fcn#results-and-models)

You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

## Results

### Japanese Society of Radiological Technology Dataset (JSRT)

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                 config                                                                                 |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/x_ray/jsrt/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_bcss-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/x_ray/jsrt/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_bcss-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/x_ray/jsrt/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_bcss-512x512.py) |

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
