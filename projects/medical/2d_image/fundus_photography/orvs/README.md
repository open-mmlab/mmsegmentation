# ORVS (Online Retinal image for Vessel Segmentation (ORVS))

## Description

This project supports **`ORVS (Online Retinal image for Vessel Segmentation (ORVS))`**, which can be downloaded from [here](https://opendatalab.org.cn/ORVS).

### Dataset Overview

The ORVS dataset is a newly established collaboration between the Department of Computer Science and the Department of Vision Science at the University of Calgary. The dataset contains 49 images collected from a clinic in Calgary, Canada, consisting of 42 training images and 7 testing images. All images were obtained using a Zeiss Visucam 200 with a 30-degree field of view (FOV). The image size is 1444×1444 pixels with 24 bits per pixel. The images are stored in JPEG format with low compression, which is common in ophthalmic practice. All images were manually traced by an expert who has been working in the field of retinal image analysis and has been trained to mark all pixels belonging to retinal vessels. The Windows Paint 3D tool was used for manual image annotation.

### Original Statistic Information

| Dataset name                                           | Anatomical region | Task type    | Modality           | Num. Classes | Train/Val/Test Images | Train/Val/Test Labeled | Release Date | License |
| ------------------------------------------------------ | ----------------- | ------------ | ------------------ | ------------ | --------------------- | ---------------------- | ------------ | ------- |
| [Bactteria detection](https://opendatalab.org.cn/ORVS) | bacteria          | segmentation | fundus photography | 2            | 130/-/72              | yes/-/yes              | 2020         | -       |

| Class Name | Num. Train | Pct. Train | Num. Val | Pct. Val | Num. Test | Pct. Test |
| :--------: | :--------: | :--------: | :------: | :------: | :-------: | :-------: |
| background |    130     |   94.83    |    -     |    -     |    72     |   94.25   |
|   vessel   |    130     |    5.17    |    -     |    -     |    72     |   5.75    |

Note:

- `Pct` means percentage of pixels in this category in all pixels.

### Visualization

![bac](https://raw.githubusercontent.com/uni-medical/medical-datasets-visualization/main/2d/semantic_seg/fundus_photography/orvs/ORVS_dataset.png)

### Prerequisites

- Python v3.8
- PyTorch v1.10.0
- [MIM](https://github.com/open-mmlab/mim) v0.3.4
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.2.0 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc5

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `orvs/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Dataset preparing

- Clone this [repository](https://github.com/AbdullahSarhan/ICPRVessels), then move `Vessels-Datasets` to `data/`.
- run script `"python tools/prepare_dataset.py"` to format data and change folder structure as below.
- run script `"python ../../tools/split_seg_dataset.py"` to split dataset and generate `train.txt`, `val.txt` and `test.txt`. If the label of official validation set and test set can't be obtained, we generate `train.txt` and `val.txt` from the training set randomly.

```none
  mmsegmentation
  ├── mmseg
  ├── projects
  │   ├── medical
  │   │   ├── 2d_image
  │   │   │   ├── fundus_photography
  │   │   │   │   ├── orvs
  │   │   │   │   │   ├── configs
  │   │   │   │   │   ├── datasets
  │   │   │   │   │   ├── tools
  │   │   │   │   │   ├── data
  │   │   │   │   │   │   ├── train.txt
  │   │   │   │   │   │   ├── test.txt
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
@inproceedings{sarhan2021transfer,
  title={Transfer learning through weighted loss function and group normalization for vessel segmentation from retinal images},
  author={Sarhan, Abdullah and Rokne, Jon and Alhajj, Reda and Crichton, Andrew},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={9211--9218},
  year={2021},
  organization={IEEE}
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
