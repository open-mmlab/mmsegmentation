# Covid-19 CT Chest X-ray

## Description

This project support **`Covid-19 CT Chest X-ray `**, and the dataset used in this project can be downloaded from [here](https://github.com/ieee8023/covid-chestxray-dataset).

### Dataset Overview

In the context of a COVID-19 pandemic, we want to improve prognostic predictions to triage and manage patient care. Data is the first step to developing any diagnostic/prognostic tool. While there exist large public datasets of more typical chest X-rays from the NIH \[Wang 2017\], Spain \[Bustos 2019\], Stanford \[Irvin 2019\], MIT \[Johnson 2019\] and Indiana University \[Demner-Fushman 2016\], there is no collection of COVID-19 chest X-rays or CT scans designed to be used for computational analysis.

The 2019 novel coronavirus (COVID-19) presents several unique features [Fang, 2020](https://pubs.rsna.org/doi/10.1148/radiol.2020200432) and [Ai 2020](https://pubs.rsna.org/doi/10.1148/radiol.2020200642). While the diagnosis is confirmed using polymerase chain reaction (PCR), infected patients with pneumonia may present on chest X-ray and computed tomography (CT) images with a pattern that is only moderately characteristic for the human eye [Ng, 2020](https://pubs.rsna.org/doi/10.1148/ryct.2020200034). In late January, a Chinese team published a paper detailing the clinical and paraclinical features of COVID-19. They reported that patients present abnormalities in chest CT images with most having bilateral involvement [Huang 2020](<https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30183-5/fulltext>). Bilateral multiple lobular and subsegmental areas of consolidation constitute the typical findings in chest CT images of intensive care unit (ICU) patients on admission [Huang 2020](<https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30183-5/fulltext>). In comparison, non-ICU patients show bilateral ground-glass opacity and subsegmental areas of consolidation in their chest CT images [Huang 2020](<https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30183-5/fulltext>). In these patients, later chest CT images display bilateral ground-glass opacity with resolved consolidation [Huang 2020](<https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30183-5/fulltext>).

### Statistic Information

| dataset name                                                           | anatomical region | task type    | modality | num. classes | train/val/test images | release date | License                                                         |
| ---------------------------------------------------------------------- | ----------------- | ------------ | -------- | ------------ | --------------------- | ------------ | --------------------------------------------------------------- |
| [Covid-19-ct-cxr](https://github.com/ieee8023/covid-chestxray-dataset) | thorax            | segmentation | x_ray    | 2            | 205/-/714             | 2021         | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

| class name | pixel percentage | Num. images in this category |
| ---------- | ---------------- | ---------------------------- |
| background | 72.84            | 205                          |
| lung       | 27.16            | 205                          |

### Visualization

![cov19ctcxr](https://github.com/uni-medical/medical-datasets-visualization/blob/main/2d/semantic_seg/x_ray/covid_19_ct_cxr/covid_19_ct_cxr_dataset.png?raw=true)

### Prerequisites

- Python 3.8
- PyTorch 1.10.0
- pillow(PIL) 9.3.0
- scikit-learn(sklearn) 1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of PYTHONPATH, which should point to the project's directory so that Python can locate the module files. In covid_19_ct_cxr/ root directory, run the following line to add the current directory to PYTHONPATH:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://github.com/ieee8023/covid-chestxray-dataset) and decompression data to path 'data/'.
- run script `"python tools/prepare_dataset.py"` to split dataset and change folder structure as below.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── x_ray
  │   │   │   │   ├── covid_19_ct_cxr
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

### Covid-19 CT Chest X-ray

|     Method      | Backbone | Crop Size |   lr   | mIoU  | mDice |                                                                                            config                                                                                            |
| :-------------: | :------: | :-------: | :----: | :---: | :---: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| fcn_unet_s5-d16 |   unet   |  512x512  |  0.01  | 76.48 | 84.68 |  [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/x_ray/covid_19_ct_cxr/configs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_covid-19-ct-cxr-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.001  | 61.06 | 63.69 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/x_ray/covid_19_ct_cxr/configs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_covid-19-ct-cxr-512x512.py)  |
| fcn_unet_s5-d16 |   unet   |  512x512  | 0.0001 | 58.87 | 62.42 | [config](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/medical/2d_image/x_ray/covid_19_ct_cxr/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_covid-19-ct-cxr-512x512.py) |

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
