# Mapillary Vistas Dataset

Support **`Mapillary Vistas Dataset`**

## Description

Author: AI-Tianlong

This project implements **`Mapillary Vistas Dataset`**

### Dataset preparing

preparing `Mapillary Vistas Dataset` dataset like this [structure](https://github.com/open-mmlab/mmsegmentation/blob/a74270e6f9554e108000a641b4a6746316d34e55/projects/Mapillary_dataset/docs/en/user_guides/2_dataset_prepare.md)👈Dataset Preparing guide

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

### Training commands with `deeplabv3plus_r101-d8_4xb2-240k_mapillay-512x1024.py`

```bash
#Dataset train commands
#at `mmsegmentation` folder
bash tools/dist_train.sh projects/Mapillary_dataset/configs/deeplabv3plus_r101-d8_4xb2-240k_mapillay-512x1024.py 4
```

## Checklist

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

  - [ ] Basic docstrings & proper citation

  - [ ] Test-time correctness

  - [ ] A full README

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

  - [ ] Unit tests

  - [ ] Code polishing

  - [ ] Metafile.yml

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
