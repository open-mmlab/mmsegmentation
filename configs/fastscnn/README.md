# Fast-SCNN for Semantic Segmentation

## Introduction
```
@article{DBLP:journals/corr/abs-1902-04502,
  author    = {Rudra P. K. Poudel and
               Stephan Liwicki and
               Roberto Cipolla},
  title     = {Fast-SCNN: Fast Semantic Segmentation Network},
  journal   = {CoRR},
  volume    = {abs/1902.04502},
  year      = {2019},
  url       = {http://arxiv.org/abs/1902.04502},
  archivePrefix = {arXiv},
  eprint    = {1902.04502},
  timestamp = {Tue, 21 May 2019 18:03:38 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1902-04502.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Results and models

### Cityscapes
|   Method   | Backbone  | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                              download                                                                                              |
|------------|-----------|-----------|--------:|----------|----------------|------:|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fast-SCNN  | Fast-SCNN | 512x1024  |   80000 |      8.4 |          63.61 | 69.06 | -             | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_4x8_80k_lr0.12_cityscapes-cae6c46a.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_4x8_80k_lr0.12_cityscapes-20200807_165744.log.json)     |
