# Kvasir-SEG Segmented Polyp Dataset from Aliyun (Kvasir SEG Aliyun)

## Description

This project supports **`Kvasir-SEG Segmented Polyp Dataset from Aliyun (Kvasir SEG Aliyun) `**, which can be downloaded from [here](https://tianchi.aliyun.com/dataset/84385).

### Dataset Overview

Colorectal cancer is the second most common cancer type among women and third most common among men. Polyps are precursors to colorectal cancer and therefore important to detect and remove at an early stage. Polyps are found in nearly half of the individuals at age 50 that undergo a colonoscopy screening, and their frequency increase with age.Polyps are abnormal tissue growth from the mucous membrane, which is lining the inside of the GI tract, and can sometimes be cancerous. Colonoscopy is the gold standard for detection and assessment of these polyps with subsequent biopsy and removal of the polyps. Early disease detection has a huge impact on survival from colorectal cancer. Increasing the detection of polyps has been shown to decrease risk of colorectal cancer. Thus, automatic detection of more polyps at an early stage can play a crucial role in prevention and survival from colorectal cancer.

The Kvasir-SEG dataset is based on the previous Kvasir dataset, which is the first multi-class dataset for gastrointestinal (GI) tract disease detection and classification. It contains annotated polyp images and their corresponding masks. The pixels depicting polyp tissue, the ROI, are represented by the foreground (white mask), while the background (in black) does not contain positive pixels. These images were collected and verified by experienced gastroenterologists from Vestre Viken Health Trust in Norway. The classes include anatomical landmarks, pathological findings and endoscopic procedures.

### Information Statistics

| Dataset Name                                           | Anatomical Region | Task Type    | Modality  | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License                                                   |
| ------------------------------------------------------ | ----------------- | ------------ | --------- | ------------ | --------------------- | ---------------------- | ------------ | --------------------------------------------------------- |
| [kvasir-seg](https://tianchi.aliyun.com/dataset/84385) | abdomen           | segmentation | endoscopy | 2            | 1000/-/-              | yes/-/-                | 2020         | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    1000    |   84.72    |    -     |    -     |     -     |     -     |
|   polyp    |    1000    |   15.28    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![kvasir_seg_aliyun](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/endoscopy_images/kvasir_seg_aliyun/kvasir_seg_aliyun_dataset.png?raw=true)

### Dataset Citation

```
@inproceedings{jha2020kvasir,
	title={Kvasir-seg: A segmented polyp dataset},
	author={Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and Lange, Thomas de and Johansen, Dag and Johansen, H{\aa}vard D},
	booktitle={International Conference on Multimedia Modeling},
	pages={451--462},
	year={2020},
	organization={Springer}
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

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `kvasir_seg_aliyun/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset Preparing

- download dataset from [here](https://tianchi.aliyun.com/dataset/84385) and decompression data to path 'data/.'.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set cannot be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

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
| background |    800     |   84.66    |   200    |  84.94   |     -     |     -     |
|   polyp    |    800     |   15.34    |   200    |  15.06   |     -     |     -     |

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
  - [ ] Test-time correctness
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
