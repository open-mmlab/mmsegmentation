# EndoVis2018RSS

## Description

This project supports **`EndoVis2018RSS`**, which can be downloaded from [here](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/).

### Dataset Overview

Our training dataset is made up of 16 robotic nephrectomy procedures recorded using da Vinci Xi systems in porcine labs. The original video data was recorded at 60 Hz and to reduce labelling cost we subsample this to 2 Hz. Sequences with little or no motion are manually removed to leave 149 frames per procedure. Video frames are 1280x1024 and we provide the left and right eye camera image as well as the stereo camera calibration parameters. Labels are only provided for the left image.

In each frame we hand label several man-made and anatomical objects. The annotations were performed by hand by several Intuitive Surgical employees with knowledge of porcine anatomy. Only a single annotator worked on each dataset. Best efforts will be made to ensure anatomical correctness of all labels however we cannot guarantee that some errors do not occur, particularly in cases where tissue type is ambiguous. The classes found in the training and test will be:

- da Vinci robotic surgical instrument parts
  - Shaft
  - Wrist
  - Jaws
- Drop in Ultrasound Probe
- Suturing Needles
- Suturing thread
- Clips/clamps
- Kidney parenchyma
  - Fascia covered
  - Uncovered
- Small bowel
- Background tissue
- Each class will have a distinct numerical label in a ground truth image. A supplied json file will contain the class name to numerical label mapping.

### Original Statistic Information

| Dataset name                                                                                     | Anatomical region | Task type    | Modality  | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                         |
| ------------------------------------------------------------------------------------------------ | ----------------- | ------------ | --------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------- |
| [EndoVis2018RSS](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/) | other             | segmentation | endoscopy | 11           | 2235/-/319            | yes/-/no               | 2018         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

|    Class Name     | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :---------------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|    background     |    2235    |   45.35    |    -     |    -     |     -     |     -     |
|  instrumentShaft  |    1012    |    8.11    |    -     |    -     |     -     |     -     |
| instrumentClasper |    268     |    0.41    |    -     |    -     |     -     |     -     |
|  instrumentWrist  |    372     |    0.50    |    -     |    -     |     -     |     -     |
| kidneyParenchyma  |    156     |    0.52    |    -     |    -     |     -     |     -     |
|   coveredKidney   |    1396    |   17.21    |    -     |    -     |     -     |     -     |
|      thread       |     12     |    0.00    |    -     |    -     |     -     |     -     |
|      clamps       |    2194    |    4.66    |    -     |    -     |     -     |     -     |
|  suturingNeedle   |    2157    |    9.42    |    -     |    -     |     -     |     -     |
| suctionInstrument |    1391    |   11.69    |    -     |    -     |     -     |     -     |
|  smallIntestine   |    1645    |    2.13    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/endoscopy/EndoVis2018RSS/EndoVis2018RSS_dataset.png)

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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `EndoVis2018RSS/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- Download dataset from [here](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/) and save it to the `data/` directory .
- Decompress data to path `data/`. This will create a new folder named `data/EndoVis2018RSS/`, which contains the original image data.
- run script `python tools/prepare_dataset.py` to format data and change folder structure as below.
- run script `python ../../tools/split_seg_dataset.py` to split dataset. For the Bacteria_detection dataset, as there is no test or validation dataset, we sample 20% samples from the whole dataset as the validation dataset and 80% samples for training data and make two filename lists `train.txt` and `val.txt`. As we set the random seed as the hard code, we eliminated the randomness, the dataset split actually can be reproducible.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── endoscopy
  │   │   │   │   ├── EndoVis2018RSS
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

|    Class Name     | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :---------------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
|    background     |    1788    |   45.43    |   447    |  45.01   |     -     |     -     |
|  instrumentShaft  |    812     |    8.02    |   200    |   8.48   |     -     |     -     |
| instrumentClasper |    218     |    0.43    |    50    |   0.36   |     -     |     -     |
|  instrumentWrist  |    294     |    0.51    |    78    |   0.47   |     -     |     -     |
| kidneyParenchyma  |    122     |    0.53    |    34    |   0.47   |     -     |     -     |
|   coveredKidney   |    1124    |   17.21    |   272    |  17.20   |     -     |     -     |
|      thread       |     10     |    0.00    |    2     |   0.00   |     -     |     -     |
|      clamps       |    1757    |    4.65    |   437    |   4.67   |     -     |     -     |
|  suturingNeedle   |    1728    |    9.46    |   429    |   9.26   |     -     |     -     |
| suctionInstrument |    1102    |   11.61    |   289    |  11.97   |     -     |     -     |
|  smallIntestine   |    1309    |    2.14    |   336    |   2.10   |     -     |     -     |

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

### EndoVis2018RSS

***Note: The following experimental results are based on the data randomly partitioned according to the above method described in the dataset preparing section.***

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                       config                                        |         download         |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :---------------------------------------------------------------------------------: | :----------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_EndoVis2018RSS-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_EndoVis2018RSS-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_EndoVis2018RSS-512x512.py) | [model](<>) \| [log](<>) |

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
