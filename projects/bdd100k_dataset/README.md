# BDD100K Dataset

Support **`BDD100K Dataset`**

## Description

Author: CastleDream

This project implements **`BDD100K Dataset`**

### Dataset preparing

Preparing `BDD100K Dataset` dataset following [BDD100K Dataset Preparing Guide](https://github.com/open-mmlab/mmsegmentation/tree/main/projects/mapillary_dataset/docs/en/user_guides/2_dataset_prepare.md#bdd100k)

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── bdd100k
│   │   ├── images
│   │   │   └── 10k
|   │   │   │   ├── test
|   │   │   │   ├── train
|   │   │   │   └── val
│   │   └── labels
│   │   │   └── sem_seg
|   │   │   │   ├── colormaps
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── masks
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── polygons
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
|   │   │   │   └── rles
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
```

### Training commands

```bash
# Dataset train commands
# at `mmsegmentation` folder
bash tools/dist_train.sh projects/mapillary_dataset/configs/deeplabv3plus_r101-d8_4xb2-240k_mapillay_v1-512x1024.py 4
```

## Thanks

- [\[Datasets\] Add Mapillary Vistas Datasets to MMSeg Core Package. #2576](https://github.com/open-mmlab/mmsegmentation/pull/2576/files)
- [\[Feature\] Support CIHP dataset #1493](https://github.com/open-mmlab/mmsegmentation/pull/1493/files)
