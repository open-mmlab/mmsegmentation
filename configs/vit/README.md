# Vision Transformer

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

## Results and models

### Cityscapes

| Method  | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                     | download                                                                                                                                                                                                                                                                                                                                               |
| ------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| UPerNet | Vit     | 512x1024  |   40000 |       |           | |          | |
| UPerNet | Deit-S  | 512x1024  |   40000 |       |           | |          | |
| UPerNet | Deit-B  | 512x1024  |   40000 |       |           | |          | |
| DeepLabV3 | Vit     | 512x1024  |   40000 |       |           | |          | |
| DeepLabV3 | Deit-S  | 512x1024  |   40000 |       |           | |          | |
| DeepLabV3 | Deit-B  | 512x1024  |   40000 |       |           | |          | |
| PSPNet | Vit     | 512x1024  |   40000 |       |           | |          | |
| PSPNet | Deit-S  | 512x1024  |   40000 |       |           | |          | |
| PSPNet | Deit-B  | 512x1024  |   40000 |       |           | |          | |
| FCN | Vit     | 512x1024  |   40000 |       |           | |          | |
| FCN | Deit-S  | 512x1024  |   40000 |       |           | |          | |
| FCN | Deit-B  | 512x1024  |   40000 |       |           | |          | |

### ADE20K

| Method  | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                 | download                                                                                                                                                                                                                                                                                                                               |
| ------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UPerNet | Vit     | 512x512  |   80000 |       |           | |          | |
| UPerNet | Deit-S  | 512x512  |   80000 |       |           | |          | |
| UPerNet | Deit-B  | 512x512  |   80000 |       |           | |          | |
| DeepLabV3 | Vit     | 512x512  |   80000 |       |           | |          | |
| DeepLabV3 | Deit-S  | 512x512  |   80000 |       |           | |          | |
| DeepLabV3 | Deit-B  | 512x512  |   80000 |       |           | |          | |
| PSPNet | Vit     | 512x512  |   80000 |       |           | |          | |
| PSPNet | Deit-S  | 512x512  |   80000 |       |           | |          | |
| PSPNet | Deit-B  | 512x512  |   80000 |       |           | |          | |
| FCN | Vit     | 512x512  |   80000 |       |           | |          | |
| FCN | Deit-S  | 512x512  |   80000 |       |           | |          | |
| FCN | Deit-B  | 512x512  |   80000 |       |           | |          | |
