# HieraSeg

Support `Deep Hierarchical Semantic Segmentation` interface on `cityscapes`

## Description

Author: AI-Tianlong

This project implements `HieraSeg` inference in the `cityscapes` dataset

## Usage

### Prerequisites

- Python 3.8
- PyTorch 1.6 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc2
- mmcv v2.0.0rc3
- mmengine
### Dataset preparing
preparing `cityscapes` dataset like this [structure](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets)  
## Testing commands
please put [`deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.pth`](https://github.com/qhanghu/HSSN_pytorch/releases/download/1.0/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.pth) to `pretrained`

#### Multi-GPUs Test  
`bash tools/dist_test.sh [configs] [model weights] [number of gpu]`  
#### For example
```shell
bash tools/dist_test.sh projects/HieraSeg_project/configs/hieraseg/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.py pretrained/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.pth 2
```

|  Dataset   | Backbone  |  decode   | Crop Size | mIoU (single scale) |                                                                        config                                                                        |                                                                model pth                                                                |
| :--------: | :-------: | :-------: | :-------: | :-----------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| Cityscapes | ResNet101 | DeeplabV3+ | 512x1024  |        81.60        | [config](https://github.com/AI-Tianlong/HSSN_pytorch/blob/main/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.py) | [github](https://github.com/qhanghu/HSSN_pytorch/releases/download/1.0/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_hiera_triplet.pth) |


## Citation

This project is modified on the basis of [qhanghu/HSSN_pytorch](https://github.com/qhanghu/HSSN_pytorch)

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
