# Fast-SCNN for Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

<a href="">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/fast_scnn.py#L272">Code Snippet</a>

<details>
<summary align="right"><a href="https://arxiv.org/abs/1902.04502">Fast-SCNN (ArXiv'2019)</a></summary>

```latex
@article{poudel2019fast,
  title={Fast-scnn: Fast semantic segmentation network},
  author={Poudel, Rudra PK and Liwicki, Stephan and Cipolla, Roberto},
  journal={arXiv preprint arXiv:1902.04502},
  year={2019}
}
```

</details>

## Results and models

### Cityscapes

| Method    | Backbone  | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                  | download                                                                                                                                                                                                                                                       |
| --------- | --------- | --------- | ------: | -------- | -------------- | ----: | ------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fast-SCNN | Fast-SCNN | 512x1024  | 160000 | 3.3 | 56.45 | 70.96 | 72.65 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853.log.json) |
