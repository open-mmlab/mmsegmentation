# HSI Drive 2.0 Dataset

Support **`HSI Drive 2.0 Dataset`**

## Description

Author: Jon Gutierrez

This project implements **`HSI Drive 2.0 Dataset`**

### Dataset preparing

Preparing `HSI Drive 2.0 Dataset` dataset following [HSI Drive 2.0 Dataset Preparing Guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#hsi-drive-2.0)

```none
mmsegmentation/data
└── HSIDrive20
    ├── images
    │   |── training []
    │   |── validation []
    │   |── test []
    └── labels
    │   |── training []
    │   |── validation []
    │   |── test []
```

### Training commands

```bash
%cd mmsegmentation
!python tools/train.py projects/hsidrive20_dataset/configs/unet-s5-d16_fcn_4xb4-160k_hsidrive-208x400.py\
--work-dir your_work_dir
```
