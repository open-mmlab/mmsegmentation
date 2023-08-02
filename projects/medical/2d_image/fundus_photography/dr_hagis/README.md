# DR HAGIS: Diabetic Retinopathy, Hypertension, Age-related macular degeneration and Glacuoma ImageS

## Description

This project supports **`DR HAGIS: Diabetic Retinopathy, Hypertension, Age-related macular degeneration and Glacuoma ImageS`**, which can be downloaded from [here](https://paperswithcode.com/dataset/dr-hagis).

### Dataset Overview

The DR HAGIS database has been created to aid the development of vessel extraction algorithms suitable for retinal screening programmes. Researchers are encouraged to test their segmentation algorithms using this database. All thirty-nine fundus images were obtained from a diabetic retinopathy screening programme in the UK. Hence, all images were taken from diabetic patients.

Besides the fundus images, the manual segmentation of the retinal surface vessels is provided by an expert grader. These manually segmented images can be used as the ground truth to compare and assess the automatic vessel extraction algorithms. Masks of the FOV are provided as well to quantify the accuracy of vessel extraction within the FOV only. The images were acquired in different screening centers, therefore reflecting the range of image resolutions, digital cameras and fundus cameras used in the clinic. The fundus images were captured using a Topcon TRC-NW6s, Topcon TRC-NW8 or a Canon CR DGi fundus camera with a horizontal 45 degree field-of-view (FOV). The images are 4752x3168 pixels, 3456x2304 pixels, 3126x2136 pixels, 2896x1944 pixels or 2816x1880 pixels in size.

### Original Statistic Information

| Dataset name                                            | Anatomical region | Task type    | Modality           | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License |
| ------------------------------------------------------- | ----------------- | ------------ | ------------------ | ------------ | --------------------- | ---------------------- | ------------ | ------- |
| [DR HAGIS](https://paperswithcode.com/dataset/dr-hagis) | head and neck     | segmentation | fundus photography | 2            | 40/-/-                | yes/-/-                | 2017         | -       |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |     40     |   96.38    |    -     |    -     |     -     |     -     |
|   vessel   |     40     |    3.62    |    -     |    -     |     -     |     -     |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/fundus_photography/dr_hagis/dr_hagis_dataset.png)

## Usage

### Prerequisites

- Python v3.8
- PyTorch v1.10.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `dr_hagis/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- download dataset from [here](https://paperswithcode.com/dataset/dr-hagis) and decompress data to path `'data/'`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set can't be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── fundus_photography
  │   │   │   │   ├── dr_hagis
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
| background |     32     |   96.21    |    8     |  97.12   |     -     |     -     |
|   vessel   |     32     |    3.79    |    8     |   2.88   |     -     |     -     |

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

## Dataset Citation

If this work is helpful for your research, please consider citing the below paper.

```
@article{holm2017dr,
  title={DR HAGIS—a fundus image database for the automatic extraction of retinal surface vessels from diabetic patients},
  author={Holm, Sven and Russell, Greg and Nourrit, Vincent and McLoughlin, Niall},
  journal={Journal of Medical Imaging},
  volume={4},
  number={1},
  pages={014503--014503},
  year={2017},
  publisher={Society of Photo-Optical Instrumentation Engineers}
}
```

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
