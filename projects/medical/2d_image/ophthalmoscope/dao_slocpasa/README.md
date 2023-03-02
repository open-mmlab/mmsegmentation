# Dataset for AO-SLO cone photoreceptor automatic segmentation and analysis

## Description

This project support **`Dataset for AO-SLO cone photoreceptor automatic segmentation and analysis `**, and the dataset used in this project can be downloaded from [here](https://people.duke.edu/~sf59/Chiu_BOE_2013_dataset.htm).

### Dataset Overview

This dataset contains 840 images (150 × 150 pixels) from the [Garrioch et al.](https://opg.optica.org/boe/fulltext.cfm?uri=boe-4-6-924&id=253777#ref44) study, where the methods for image acquisition and pre-processing are described in detail. To summarize, the right eye of 21 subjects (25.9 ± 6.5 years in age, 1 subject with deuteranopia) was imaged using a previously described AOSLO system with a 775 nm super luminescent diode and a 0.96 × 0.96° field of view. Four locations 0.65° from the center of fixation (bottom left, bottom right, top left, and top right) were imaged, capturing 150 frames at each site. This process was repeated 10 times for each subject. Axial length measurements were also acquired with an IOL Master (Carl Zeiss Meditec, Dublin, CA) to determine the lateral resolution of the captured images.

### Statistic Information

| dataset name                                                            | anatomical region | task type    | modality       | num. classes | train/val/test images | release date | License                                                         |
| ----------------------------------------------------------------------- | ----------------- | ------------ | -------------- | ------------ | --------------------- | ------------ | --------------------------------------------------------------- |
| [Dao-slocpasa](https://people.duke.edu/~sf59/Chiu_BOE_2013_dataset.htm) | head_and_neck     | segmentation | ophthalmoscope | 2            | 840/-/-               | 2012         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| class name     | pixel percentage | Num. images in this category |
| -------------- | ---------------- | ---------------------------- |
| background     | 44.47            | 840                          |
| photoreceptors | 55.53            | 840                          |

### Visualization

![daoslocpasa](https://github.com/uni-medical/medical-datasets-visualization/blob/main/2d/semantic_seg/ophthalmoscope/dao_slocpasa/dao_slocpasa_dataset.png?raw=true)

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- pillow(PIL) 9.3.0
- scikit-learn(sklearn) 1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of PYTHONPATH, which should point to the project's directory so that Python can locate the module files. In dao_slocpasa/ root directory, run the following line to add the current directory to PYTHONPATH:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://people.duke.edu/~sf59/Chiu_BOE_2013_dataset.htm) and decompression data to path 'data/'.
- run script `"python tools/prepare_dataset.py"` to split dataset and change folder structure as below.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── ophthalmoscope
  │   │   │   │   ├── dao_slocpasa
  │   │   │   │   │   ├── configs
  │   │   │   │   │   ├── datasets
  │   │   │   │   │   ├── tools
  │   │   │   │   │   ├── data
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
  │   │   │   │   │   │   │   ├── val
  │   │   │   │   |   │   │   │   ├── yyy.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── yyy.png
```

### Training commands

```shell
mim train mmseg ./configs/${CONFIG_PATH}
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```shell
mim train mmseg ./configs/${CONFIG_PATH}  --launcher pytorch --gpus 8
```

### Testing commands

```shell
mim test mmseg ./configs/${CONFIG_PATH}  --checkpoint ${CHECKPOINT_PATH}
```

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/configs/fcn#results-and-models)

You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

## Results

### Dataset for AO-SLO cone photoreceptor automatic segmentation and analysis

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                             config                                                                                              |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/ophthalmoscope/dao_slocpasa/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_dao-slocpasa-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/ophthalmoscope/dao_slocpasa/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_dao-slocpasa-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/ophthalmoscope/dao_slocpasa/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_dao-slocpasa-512x512.py) |

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
