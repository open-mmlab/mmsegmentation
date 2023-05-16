# Mapillary Vistas Dataset

Support **`Mapillary Vistas Dataset`**

## Description

Author: AI-Tianlong

This project implements **`Mapillary Vistas Dataset`**

### Dataset preparing

Preparing `Mapillary Vistas Dataset` dataset following [Mapillary Vistas Dataset Preparing Guide](https://github.com/open-mmlab/mmsegmentation/tree/main/projects/mapillary_dataset/docs/en/user_guides/2_dataset_prepare.md)

```none
  mmsegmentation
  ├── mmseg
  ├── tools
  ├── configs
  ├── data
  │   ├── mapillary
  │   │   ├── training
  │   │   │   ├── images
  │   │   │   ├── v1.2
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── labels_mask
  |   │   │   │   └── panoptic
  │   │   │   ├── v2.0
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── labels_mask
  |   │   │   │   ├── panoptic
  |   │   │   │   └── polygons
  │   │   ├── validation
  │   │   │   ├── images
  │   │   │   ├── v1.2
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── labels_mask
  |   │   │   │   └── panoptic
  │   │   │   ├── v2.0
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── labels_mask
  |   │   │   │   ├── panoptic
  |   │   │   │   └── polygons
```

### Training commands

```bash
# Dataset train commands
# at `mmsegmentation` folder
bash tools/dist_train.sh projects/mapillary_dataset/configs/deeplabv3plus_r101-d8_4xb2-240k_mapillay_v1-512x1024.py 4
```

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

  - [x] Basic docstrings & proper citation

  - [ ] Test-time correctness

  - [x] A full README

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

- [x] Milestone 3: Good to be a part of our core package!

  - [x] Type hints and docstrings

  - [x] Unit tests

  - [x] Code polishing

  - [x] Metafile.yml

- [x] Move your modules into the core package following the codebase's file hierarchy structure.

- [x] Refactor your modules into the core package following the codebase's file hierarchy structure.
