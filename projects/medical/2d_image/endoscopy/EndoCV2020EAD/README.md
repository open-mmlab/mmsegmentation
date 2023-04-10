# EndoCV2020EAD

## Description

This project supports **`EndoCV2020EAD`**, which can be downloaded from [here](https://ead2020.grand-challenge.org/Data/).

### Dataset Overview

Endoscopy is a widely used clinical procedure for the early detection of numerous cancers (e.g., nasopharyngeal, oesophageal adenocarcinoma, gastric, colorectal cancers, bladder cancer etc.), therapeutic procedures and minimally invasive surgery (e.g.,laparoscopy). During this procedure an endoscope is used; a long, thin, rigid or flexible tube having a light source and a camera at the tip which allows to visualise inside of affected organs on a screen. A major drawback of these video frames is that they are heavily corrupted with multiple artefacts (e.g., pixel saturations, motion blur, defocus, specular reflections, bubbles, fluid, debris etc.). These artefacts not only present difficulty in visualising the underlying tissue during diagnosis but also affect any postanalysis methods required for follow-ups (e.g., video mosaicking done for follow-ups and archival purposes, and video-frame retrieval needed for reporting).

Accurate detection of artefacts is a core challenge in a wide-range of endoscopic applications addressing multiple different disease areas. The importance of precise detection of these artefacts is essential for high-quality endoscopic frame restoration and crucial for realising reliable computer assisted endoscopy tools for improved patient care. Existing endoscopy workflows detect only one artefact class which is insufficient to obtain high-quality frame restoration. In general, the same video frame can be corrupted with multiple artefacts, e.g. motion blur, specular reflections, and low contrast can be present in the same frame. Further, not all artefact types contaminate the frame equally. So, unless multiple artefacts present in the frame are known with their precise spatial location, clinically relevant frame restoration quality cannot be guaranteed. Another advantage of such detection is that frame quality assessments can be guided to minimise the number of frames that gets discarded during automated video analysis.The aim of this task is to localise bounding boxes, predict class labels and pixel-wise segmentation of 8 different artefact classes for given frames and clinical endoscopy video clips.

The 8 classes in this challenge include specularity, bubbles, saturation, contrast, blood, instrument, blur and imaging artefacts. The algorithms are also evaluated on generalisation of detection methods used in this category.

### Original Statistic Information

| Dataset name                                               | Anatomical region | Task type    | Modality  | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                         |
| ---------------------------------------------------------- | ----------------- | ------------ | --------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------- |
| [EndoCV2020EAD](https://ead2020.grand-challenge.org/Data/) | endoscopy         | segmentation | endoscopy | 2            | 366/-/-               | yes/-/-                | 2020         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    573     |   96.58    |    -     |    -     |     -     |     -     |
| artefacts  |    347     |    3.42    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/endoscopy/EndoCV2020EAD/EndoCV2020EAD_dataset.png)

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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `EndoCV2020EAD/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- Download dataset from [here](https://ead2020.grand-challenge.org/Data/) and save it to the `data/` directory .
- Decompress data to path `data/`. This will create a new folder named `data/EndoCV2020EAD/`, which contains the original image data.
- run script `python tools/prepare_dataset.py` to format data and change folder structure as below.
- run script `python ../../tools/split_seg_dataset.py` to split dataset. For the Bacteria_detection dataset, as there is no test or validation dataset, we sample 20% samples from the whole dataset as the validation dataset and 80% samples for training data and make two filename lists `train.txt` and `val.txt`. As we set the random seed as the hard code, we eliminated the randomness, the dataset split actually can be reproducible.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── endoscopy
  │   │   │   │   ├── EndoCV2020EAD
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
| background |    458     |   96.83    |   115    |  95.53   |     -     |     -     |
| artefacts  |    273     |    3.17    |    74    |   4.47   |     -     |     -     |

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

### EndoCV2020EAD

***Note: The following experimental results are based on the data randomly partitioned according to the above method described in the dataset preparing section.***

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                       config                                       |         download         |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :--------------------------------------------------------------------------------: | :----------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_EndoCV2020EAD-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_EndoCV2020EAD-512x512.py)  | [model](<>) \| [log](<>) |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](./configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_EndoCV2020EAD-512x512.py) | [model](<>) \| [log](<>) |

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
