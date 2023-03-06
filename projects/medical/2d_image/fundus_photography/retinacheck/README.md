# IOSTAR Retinal Vessel Segmentation

## Description

This project support **`IOSTAR Retinal Vessel Segmentation`**, and the dataset used in this project can be downloaded from [here](http://www.retinacheck.org/download-iostar-retinal-vessel-segmentation-dataset).

### Dataset Overview

This is an improved version of our vessel segmentation dataset on Scanning Laser Ophthalmoscopy (SLO) images. The images in the IOSTAR vessel segmentation dataset are acquired with an EasyScan camera (i-Optics Inc., the Netherlands), which is based on a SLO technique with a 45 degree Field of View (FOV).  The IOSTAR vessel segmentation dataset includes 30 images with a resolution of 1024 × 1024 pixels. All the vessels in this dataset are annotated by a group of experts working in the field of retinal image analysis.

### Original Statistic Information

| Dataset name | Anatomical region | Task type | Modality | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled |  Release Date | License |
| - | - | - | - | - | - | - | - | - |
| [RetinaCheck](http://www.retinacheck.org/download-iostar-retinal-vessel-segmentation-dataset) | eye | segmentation | fundus photophy | 2 | 20/-/20 | yes/-/- | 2020 | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |     30     |   92.23    |    -     |    -     |     -     |     -     |
|   vessel   |     30     |    7.77    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/fundus_photography/retinacheck/retinacheck_dataset.png)

## Dataset Citation
```bibtex
@article{zhang2016robust,
  title={Robust retinal vessel segmentation via locally adaptive derivative frames in orientation scores},
  author={Zhang, Jiong and Dashtbozorg, Behdad and Bekkers, Erik and Pluim, Josien PW and Duits, Remco and ter Haar Romeny, Bart M},
  journal={IEEE transactions on medical imaging},
  volume={35},
  number={12},
  pages={2631--2644},
  year={2016},
  publisher={IEEE}
}

@inproceedings{abbasi2015biologically,
  title={Biologically-inspired supervised vasculature segmentation in SLO retinal fundus images},
  author={Abbasi-Sureshjani, Samaneh and Smit-Ockeloen, Iris and Zhang, Jiong and Ter Haar Romeny, Bart},
  booktitle={Image Analysis and Recognition: 12th International Conference, ICIAR 2015, Niagara Falls, ON, Canada, July 22-24, 2015, Proceedings 12},
  pages={325--334},
  year={2015},
  organization={Springer}
}
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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `retinacheck/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](http://www.retinacheck.org/download-iostar-retinal-vessel-segmentation-dataset) and decompression data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to split dataset and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set can't be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── fundus_photography
  │   │   │   │   ├── retinacheck
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
| background |     24     |   92.23    |    6     |  92.25   |     -     |     -     |
|   vessel   |     24     |    7.77    |    6     |   7.75   |     -     |     -     |

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

### Retina Check

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                        config                                                                                         |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_retinacheck-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_retinacheck-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/fundus_photography/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_retinacheck-512x512.py) |

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
