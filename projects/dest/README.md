# DEST

[DEST: Depth Estimation with Simplified Transformer](https://arxiv.org/abs/2204.13791)

## Description

Transformer and its variants have shown state-of-the-art results in many vision tasks recently, ranging from image classification to dense prediction. Despite of their success, limited work has been reported on improving the model efficiency for deployment in latency-critical applications, such as autonomous driving and robotic navigation. In this paper, we aim at improving upon the existing transformers in vision, and propose a method for Dense Estimation with Simplified Transformer (DEST), which is efficient and particularly suitable for deployment on GPU-based platforms. Through strategic design choices, our model leads to significant reduction in model size, complexity, as well as inference latency, while achieving superior accuracy as compared to state-of-the-art in the task of self-supervised monocular depth estimation. We also show that our design generalize well to other dense prediction task such as semantic segmentation without bells and whistles.

## Usage

### Prerequisites

- Python 3.8.12
- PyTorch 1.11
- mmcv v1.7.0
- Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) from source

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the mmsegmentaions directory so that Python can locate the configuration files in mmsegmentation.

### Dataset preparing

Preparing `cityscapes` dataset following this [Dataset Preparing Guide](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets)

### Training commands

```shell
mim train mmsegmentation projects/dest/configs/dest_simpatt-b0_1024x1024_160k_cityscapes.py --work-dir work_dirs/dest
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```shell
mim train mmsegmentation projects/dest/configs/dest_simpatt-b0_1024x1024_160k_cityscapes.py --work-dir work_dirs/dest --launcher pytorch --gpus 8
```

### Testing commands

```shell
mim test mmsegmentation projects/dest/configs/dest_simpatt-b0_1024x1024_160k_cityscapes.py --work-dir work_dirs/dest --checkpoint ${CHECKPOINT_PATH} --eval mIoU
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                       | download                                                                                                                                                                                                                                                                                                                                                      |
| ------ | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DEST   | SMIT-B0  | 1024x1024 |  160000 |        - | -              | 64.34 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dest/dest_simpatt-b0_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b0_1024x1024_160k_cityscapes_20230105_232025-11f73f34.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b0_1024x1024_160k_cityscapes_20230105_232025.log)                                                                                       |
| DEST   | SMIT-B1  | 1024x1024 |  160000 |        - | -              | 68.21 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dest/dest_simpatt-b1_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b1_1024x1024_160k_cityscapes_20230105_232358-0dd4e86e.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b1_1024x1024_160k_cityscapes_20230105_232358.logmmsegmentation/v0.5/dest/dest_simpatt-b1_1024x1024_160k_cityscapes_20230105_232358.log) |
| DEST   | SMIT-B2  | 1024x1024 |  160000 |        - | -              | 71.89 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dest/dest_simpatt-b2_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b2_1024x1024_160k_cityscapes_20230105_231943-b06319ae.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b2_1024x1024_160k_cityscapes_20230105_231943.log)                                                                                       |
| DEST   | SMIT-B3  | 1024x1024 |  160000 |        - | -              | 73.51 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dest/dest_simpatt-b3_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b3_1024x1024_160k_cityscapes_20230105_231800-ee4cec5c.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b3_1024x1024_160k_cityscapes_20230105_231800.log)                                                                                       |
| DEST   | SMIT-B4  | 1024x1024 |  160000 |        - | -              | 73.99 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dest/dest_simpatt-b4_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b4_1024x1024_160k_cityscapes_20230105_232155-3ca9f4fc.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b4_1024x1024_160k_cityscapes_20230105_232155.log)                                                                                       |
| DEST   | SMIT-B5  | 1024x1024 |  160000 |        - | -              | 75.28 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dest/dest_simpatt-b5_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b5_1024x1024_160k_cityscapes_20230105_231411-e83819b5.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dest/dest_simpatt-b5_1024x1024_160k_cityscapes_20230105_231411.log)                                                                                       |

Note:

- The above models are all training from scratch without pretrained backbones. Accuracy can be further enhanced by appropriate pretraining.
- Training of DEST is not very stable, which is sensitive to random seeds.

## Citation

```bibtex
@article{YangDEST,
  title={Depth Estimation with Simplified Transformer},
  author={Yang, John and An, Le and Dixit, Anurag and Koo, Jinkyu and Park, Su Inn},
  journal={arXiv preprint arXiv:2204.13791},
  year={2022}
}
```

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

  - [x] Basic docstrings & proper citation

  - [x] Test-time correctness

  - [x] A full README

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

  - [ ] Unit tests

  - [ ] Code polishing

  - [ ] Metafile.yml

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
