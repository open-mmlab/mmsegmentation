# Vision Transformer

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{dosoViTskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={DosoViTskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

## Results and models

### Cityscapes

| Method  | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                     | download                                                                                                                                                                                                                                                                                                                                               |
| ------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| UPerNet | ViT-B     | 512x1024  |   40000 |       |           | 72.61 |     | |
| UPerNet | DeiT-S  | 512x1024  |   40000 |       |           | 69.28 |     | |
| UPerNet | DeiT-B  | 512x1024  |   40000 |       |           | 73.35 |    | |

### ADE20K

| Method  | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                 | download                                                                                                                                                                                                                                                                                                                               |
| ------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UPerNet | ViT-B     | 512x512  |   80000 |       |           |45.99  |    | |
| UPerNet | ViT-B     | 512x512  |   160000 |       |           |45.88  |    | |
| UPerNet | DeiT-S  | 512x512  |   80000 |       |           |41.32  |    | |
| UPerNet | DeiT-S  | 512x512  |   160000 |       |           |  |    | |
| UPerNet | DeiT-B  | 512x512  |   80000 |       |           |   |   | |
| UPerNet | DeiT-B  | 512x512  |   160000 |       |           | |   | |
