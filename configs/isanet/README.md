# Interlaced Sparse Self-Attention for Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

```
@article{huang2019isa,
  title={Interlaced Sparse Self-Attention for Semantic Segmentation},
  author={Huang Lang and Yuan Yuhui and Guo Jianyuan and Zhang Chao and Chen Xilin and Wang Jingdong},
  journal={arXiv preprint arXiv:1907.12273},
  year={2019}
}

The technical report above is also presented at:
@article{yuan2021ocnet,
  title={OCNet: Object Context for Semantic Segmentation},
  author={Yuan, Yuhui and Huang, Lang and Guo, Jianyuan and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  journal={International Journal of Computer Vision},
  pages={1--24},
  year={2021},
  publisher={Springer}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                                 download                                                                                                                                                                                                 |
|--------|----------|-----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ISANet | R-101-D8 | 512x1024  |   8 | 40000 |       |       | 79.32|          |[model](https://drive.google.com/file/d/1oWAcDwj_ILRvwWp-bcGKJ7g1QlrT7Myo/view?usp=sharing)/[log](https://drive.google.com/file/d/1oWPEtE16FYF4P4LMl1uf_qiElPaOJ0-y/view?usp=sharing)    |
| ISANet | R-101-D8 | 512x1024  |  16 |  40000 |       |       |  79.56  |      |      |
| ISANet | R-101-D8 | 512x1024  |  8 |  80000 |       |       |  79.67  |      |      |
| ISANet | R-101-D8 | 512x1024  |  16 |  80000 |       |       |  80.18  |      |      |
| NonLocal | R-101-D8 | 512x1024  | 8 |  40000 |     10.9 |           1.95 | 78.66 |
| NonLocal | R-101-D8 | 512x1024  | 8 |  80000 | -        | -              | 78.93 |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                         download                                                                                                                                                                                         |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ISANet | R-50-D8  | 512x512   |   80000 |       |       |      |          |      |
| ISANet | R-101-D8 | 512x512   |   80000 |       |       |      |          |      |
| ISANet | R-50-D8  | 512x512   |  160000 |       |       |      |          |      |
| ISANet | R-101-D8 | 512x512   |  160000 |       |       | 43.77|         |      |
| NonLocal | R-101-D8 | 512x512   |  160000 | -        | -              | 43.36 |      |      |

### Pascal VOC 2012 + Aug

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                           download                                                                                                                                                                                           |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ISANet | R-50-D8  | 512x512   |   20000 |       |       |      |          |      |
| ISANet | R-101-D8 | 512x512   |   20000 |       |       |      |          |      |
| ISANet | R-50-D8  | 512x512   |   40000 |       |       |      |          |      |
| ISANet | R-101-D8 | 512x512   |   40000 |       |       |      |          |      |
