# ISNet

[ISNet: Integrate Image-Level and Semantic-Level Context for Semantic Segmentation](https://arxiv.org/pdf/2108.12382.pdf)

## Description

This is an implementation of [ISNet](https://arxiv.org/pdf/2108.12382.pdf).
[Official Repo](https://github.com/SegmentationBLWX/sssegmentation)

## Usage

### Prerequisites

- Python 3.7
- PyTorch 1.6 or higher
- [MIM](https://github.com/open-mmlab/mim) v0.33 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc2 or higher

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `example_project/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Training commands

```shell
mim train mmsegmentation configs/isnet_r50-d8_8xb2-160k_cityscapes-512x1024.py --work-dir work_dirs/isnet
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```shell
mim train mmsegmentation configs/isnet_r50-d8_8xb2-160k_cityscapes-512x1024.py --work-dir work_dirs/isnet --launcher pytorch --gpus 8
```

### Testing commands

```shell
mim test mmsegmentation configs/isnet_r50-d8_8xb2-160k_cityscapes-512x1024.py --work-dir work_dirs/isnet --checkpoint ${CHECKPOINT_PATH}
```

| Method                                                                                                     | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                          | download                                                                        |
| ---------------------------------------------------------------------------------------------------------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | --------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| ISNet                                                                                                      | R-50-D8  | 512x1024  |       - | -        | -              | 79.32 |         80.88 | [config](configs/isnet_r50-d8_8xb2-160k_cityscapes-512x1024.py) | \[model\](https://download.openmmlab.com/mmsegmentation/v0.5/isnet/isne_xxx.pth |
| ) \| \[log\](https://download.openmmlab.com/mmsegmentation/v0.5/isnet/isnet_r50-d8_cityscapes-512x1024.pth |          |           |         |          |                |       |               |                                                                 |                                                                                 |
| )                                                                                                          |          |           |         |          |                |       |               |                                                                 |                                                                                 |

## Citation

```bibtex
@article{Jin2021ISNetII,
  title={ISNet: Integrate Image-Level and Semantic-Level Context for Semantic Segmentation},
  author={Zhenchao Jin and B. Liu and Qi Chu and Nenghai Yu},
  journal={2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021},
  pages={7169-7178}
}
```
