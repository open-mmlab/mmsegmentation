# HieraSeg

Support `Deep Hierarchical Semantic Segmentation` interface on `cityscapes`

## Description

Author: AI-Tianlong

This project implements `HieraSeg` inference in the `cityscapes` dataset

## Usage

### Prerequisites

- Python 3.8
- PyTorch 1.6 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc3
- mmcv v2.0.0rc3
- mmengine

### Dataset preparing

preparing `cityscapes` dataset like this [structure](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets)

## Testing commands

please put [`deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.pth`](https://github.com/qhanghu/HSSN_pytorch/releases/download/1.0/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.pth) to `mmsegmentation/pretrained`

#### Multi-GPUs Test

`bash tools/dist_test.sh [configs] [model weights] [number of gpu]  --tta `(--tta optional, multi-scale test, need mmengine >=0.4.0)

#### For example

```shell
bash tools/dist_test.sh projects/HieraSeg_project/configs/hieraseg/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.py pretrained/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.pth 2 --tta
```

## Results

### Cityscapes

|   Method   | Backbone | Crop Size | mIoU  | mIoU (ms+flip) |                                                                                               config                                                                                                |                                                                model pth                                                                |
| :--------: | :------: | :-------: | :---: | :------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| DeeplabV3+ | R-101-D8 | 512x1024  | 81.61 |     82.71      | [config](https://github.com/AI-Tianlong/mmsegmentation/blob/AI-Tianlong/Support_HieraSeg/projects/HieraSeg_project/configs/hieraseg/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.py) | [github](https://github.com/qhanghu/HSSN_pytorch/releases/download/1.0/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.pth) |

<img src="https://user-images.githubusercontent.com/50650583/210488953-e3e35ade-1132-47e1-9dfd-cf12b357ae80.png" width="50%"><img src="https://user-images.githubusercontent.com/50650583/210489746-e35ee229-3234-4292-a649-a8cd85f312ad.png" width="50%">

## Citation

This project is modified from [qhanghu/HSSN_pytorch](https://github.com/qhanghu/HSSN_pytorch)

```bibtex
@article{li2022deep,
  title={Deep Hierarchical Semantic Segmentation},
  author={Li, Liulei and Zhou, Tianfei and Wang, Wenguan and Li, Jianwu and Yang, Yi},
  journal={CVPR},
  year={2022}
}
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
