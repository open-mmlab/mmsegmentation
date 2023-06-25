# Covid-19 CT Chest X-ray Dataset

## Description

This project supports **`Covid-19 CT Chest X-ray Dataset`**, which can be downloaded from [here](https://github.com/ieee8023/covid-chestxray-dataset).

### Dataset Overview

In the context of a COVID-19 pandemic, we want to improve prognostic predictions to triage and manage patient care. Data is the first step to developing any diagnostic/prognostic tool. While there exist large public datasets of more typical chest X-rays from the NIH \[Wang 2017\], Spain \[Bustos 2019\], Stanford \[Irvin 2019\], MIT \[Johnson 2019\] and Indiana University \[Demner-Fushman 2016\], there is no collection of COVID-19 chest X-rays or CT scans designed to be used for computational analysis.

The 2019 novel coronavirus (COVID-19) presents several unique features [Fang, 2020](https://pubs.rsna.org/doi/10.1148/radiol.2020200432) and [Ai 2020](https://pubs.rsna.org/doi/10.1148/radiol.2020200642). While the diagnosis is confirmed using polymerase chain reaction (PCR), infected patients with pneumonia may present on chest X-ray and computed tomography (CT) images with a pattern that is only moderately characteristic for the human eye [Ng, 2020](https://pubs.rsna.org/doi/10.1148/ryct.2020200034). In late January, a Chinese team published a paper detailing the clinical and paraclinical features of COVID-19. They reported that patients present abnormalities in chest CT images with most having bilateral involvement [Huang 2020](<https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30183-5/fulltext>). Bilateral multiple lobular and subsegmental areas of consolidation constitute the typical findings in chest CT images of intensive care unit (ICU) patients on admission [Huang 2020](<https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30183-5/fulltext>). In comparison, non-ICU patients show bilateral ground-glass opacity and subsegmental areas of consolidation in their chest CT images [Huang 2020](<https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30183-5/fulltext>). In these patients, later chest CT images display bilateral ground-glass opacity with resolved consolidation [Huang 2020](<https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30183-5/fulltext>).

### Statistic Information

| Dataset Name                                                           | Anatomical Region | Task Type    | Modality | Nnum. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release date | License                                                               |
| ---------------------------------------------------------------------- | ----------------- | ------------ | -------- | ------------- | --------------------- | ---------------------- | ------------ | --------------------------------------------------------------------- |
| [Covid-19-ct-cxr](https://github.com/ieee8023/covid-chestxray-dataset) | thorax            | segmentation | x_ray    | 2             | 205/-/714             | yes/-/no               | 2021         | [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    205     |   72.84    |    -     |    -     |     -     |     -     |
|    lung    |    205     |   27.16    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![cov19ctcxr](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/x_ray/covid_19_ct_cxr/covid_19_ct_cxr_dataset.png?raw=true)

### Dataset Citation

```
@article{cohen2020covidProspective,
  title={{COVID-19} Image Data Collection: Prospective Predictions Are the Future},
  author={Joseph Paul Cohen and Paul Morrison and Lan Dao and Karsten Roth and Tim Q Duong and Marzyeh Ghassemi},
  journal={arXiv 2006.11988},
  year={2020}
}

@article{cohen2020covid,
  title={COVID-19 image data collection},
  author={Joseph Paul Cohen and Paul Morrison and Lan Dao},
  journal={arXiv 2003.11597},
  year={2020}
}
```

### Prerequisites

- Python v3.8
- PyTorch v1.10.0
- pillow(PIL) v9.3.0 9.3.0
- scikit-learn(sklearn) v1.2.0 1.2.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `covid_19_ct_cxr/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- download dataset from [here](https://github.com/ieee8023/covid-chestxray-dataset) and decompress data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set cannot be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```shell
mkdir data && cd data
git clone git@github.com:ieee8023/covid-chestxray-dataset.git
cd ..
python tools/prepare_dataset.py
python ../../tools/split_seg_dataset.py
```

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
| background |    164     |   72.88    |    41    |  72.69   |     -     |     -     |
|    lung    |    164     |   27.12    |    41    |  27.31   |     -     |     -     |

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
