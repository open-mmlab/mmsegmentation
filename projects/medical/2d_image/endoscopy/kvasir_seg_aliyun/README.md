# Kvasir-SEG Segmented Polyp Dataset from Aliyun (Kvasir SEG Aliyun)

## Description

This project support **`Kvasir-SEG Segmented Polyp Dataset from Aliyun (Kvasir SEG Aliyun) `**, and the dataset used in this project can be downloaded from [here](https://tianchi.aliyun.com/dataset/84385).

### Dataset Overview

Colorectal cancer is the second most common cancer type among women and third most common among men. Polyps are precursors to colorectal cancer and therefore important to detect and remove at an early stage. Polyps are found in nearly half of the individuals at age 50 that undergo a colonoscopy screening, and their frequency increase with age.Polyps are abnormal tissue growth from the mucous membrane, which is lining the inside of the GI tract, and can sometimes be cancerous. Colonoscopy is the gold standard for detection and assessment of these polyps with subsequent biopsy and removal of the polyps. Early disease detection has a huge impact on survival from colorectal cancer. Increasing the detection of polyps has been shown to decrease risk of colorectal cancer. Thus, automatic detection of more polyps at an early stage can play a crucial role in prevention and survival from colorectal cancer.

The Kvasir-SEG dataset is based on the previous Kvasir dataset, which is the first multi-class dataset for gastrointestinal (GI) tract disease detection and classification. It contains annotated polyp images and their corresponding masks. The pixels depicting polyp tissue, the ROI, are represented by the foreground (white mask), while the background (in black) does not contain positive pixels. These images were collected and verified by experienced gastroenterologists from Vestre Viken Health Trust in Norway. The classes include anatomical landmarks, pathological findings and endoscopic procedures.

### Information Statistics

| dataset_name                                           | anatomical region | task type    | modality  | number of categories | train/val/test image | release date | License                                                   |
| ------------------------------------------------------ | ----------------- | ------------ | --------- | -------------------- | -------------------- | ------------ | --------------------------------------------------------- |
| [kvasir-seg](https://tianchi.aliyun.com/dataset/84385) | abdomen           | segmentation | endoscopy | 2                    | 1000/-/-             | 2020         | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

| class_name | Percentage of pixels（%） | Number of pictures in this category |
| ---------- | ------------------------- | ----------------------------------- |
| background | 84.72                     | 1000                                |
| polyp      | 15.28                     | 1000                                |

### Visualization

![kvasir_seg_aliyun](https://github.com/uni-medical/medical-datasets-visualization/blob/main/2d/semantic_seg/endoscopy_images/kvasir_seg_aliyun/kvasir_seg_aliyun_dataset.png?raw=true)

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- pillow(PIL)
- scikit-learn(sklearn)
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of PYTHONPATH, which should point to the project's directory so that Python can locate the module files. In kvasir_seg_aliyun/ root directory, run the following line to add the current directory to PYTHONPATH:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://tianchi.aliyun.com/dataset/84385) and decompression data to path 'data/.'.
- run script `"python tools/prepare_dataset.py"` to split dataset and change folder structure as below.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── endoscopy
  │   │   │   │   ├── kvasir_seg_aliyun
  │   │   │   │   │   ├── configs
  │   │   │   │   │   ├── datasets
  │   │   │   │   │   ├── tools
  │   │   │   │   │   ├── data
  │   │   │   │   │   │   ├── images
  │   │   │   │   │   │   │   ├── train
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png
  │   │   │   │   │   │   │   ├── val
  │   │   │   │   |   │   │   │   ├── yyy.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── yyy.png
  │   │   │   │   │   │   ├── masks
  │   │   │   │   │   │   │   ├── train
  │   │   │   │   |   │   │   │   ├── xxx.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── xxx.png
  │   │   │   │   │   │   │   ├── val
  │   │   │   │   |   │   │   │   ├── yyy.png
  │   │   │   │   |   │   │   │   ├── ...
  │   │   │   │   |   │   │   │   └── yyy.png
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

### Kvasir-SEG Segmented Polyp Dataset from Aliyun (Kvasir SEG Aliyun)

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                                config                                                                                                |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/endoscopy/kvasir_seg_aliyun/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_kvasir-seg-aliyun-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/endoscopy/kvasir_seg_aliyun/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_kvasir-seg-aliyun-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/endoscopy/kvasir_seg_aliyun/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_kvasir-seg-aliyun-512x512.py) |

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
