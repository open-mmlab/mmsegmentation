# BDD100K Dataset

Support **`BDD100K Dataset`**

## Description

Author: CastleDream

This project implements **`BDD100K Dataset`**

### Dataset preparing

Preparing `BDD100K Dataset` dataset following [BDD100K Dataset Preparing Guide](https://github.com/open-mmlab/mmsegmentation/tree/main/projects/mapillary_dataset/docs/en/user_guides/2_dataset_prepare.md#bdd100k)

```none
mmsegmentation/data
└── bdd100k
    ├── images
    │   └── 10k
    │       ├── test [2000 entries exceeds filelimit, not opening dir]
    │       ├── train [7000 entries exceeds filelimit, not opening dir]
    │       └── val [1000 entries exceeds filelimit, not opening dir]
    └── labels
        └── sem_seg
            ├── colormaps
            │   ├── train [7000 entries exceeds filelimit, not opening dir]
            │   └── val [1000 entries exceeds filelimit, not opening dir]
            ├── masks
            │   ├── train [7000 entries exceeds filelimit, not opening dir]
            │   └── val [1000 entries exceeds filelimit, not opening dir]
            ├── polygons
            │   ├── sem_seg_train.json
            │   └── sem_seg_val.json
            └── rles
                ├── sem_seg_train.json
                └── sem_seg_val.json
```

### Training commands

```bash
%cd mmsegmentation
!python tools/train.py projects/bdd100k_dataset/configs/pspnet_r50-d8_4xb2-80k_bdd100k-512x1024.py\
--work-dir your_work_dir
```

## Thanks

- [\[Datasets\] Add Mapillary Vistas Datasets to MMSeg Core Package. #2576](https://github.com/open-mmlab/mmsegmentation/pull/2576/files)
- [\[Feature\] Support CIHP dataset #1493](https://github.com/open-mmlab/mmsegmentation/pull/1493/files)
