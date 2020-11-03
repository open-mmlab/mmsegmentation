# CGNet: A Light-weight Context Guided Network for Semantic Segmentation

## Introduction

```latext
@article{wu2019cgnet,
  title={CGNet: A Light-weight Context Guided Network for Semantic Segmentation},
  author={Tianyi Wu, Sheng Tang, Rui Zhang and Yongdong Zhang},
  journal={arXiv preprint arXiv:1811.08201},
  year={2019}
}
```

## Results and models

### Cityscapes

|  Method   | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                                          download                                                                                                                                                                                                          |
|-----------|----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CGNet | M3N21  | 680x680  |   60000 |      7.5 |           16.03 | 65.63 |     68.04 | [model]() &#124; [log]() |
| CGNet | M3N21  | 512x1024 |   60000 |      8.3 |           15.75 | 68.27 |     70.33 | [model]() &#124; [log]() |
