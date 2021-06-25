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

### ADE20K

| Method  | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                 | download                                                                                                                                                                                                                                                                                                                               |
| ------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UPerNet | ViT-B         | 512x512  | 80000  | |  | 47.71 |  |  |  |
| UPerNet | ViT-B         | 512x512  | 160000 | |  | train |  |  |  |
| UPerNet | ViT-B + neck  | 512x512  | 160000 | |  | train |  |  |  |
| UPerNet | ViT-B + norm  | 512x512  | 160000 | |  | 47.73 |  |  |  |
| UPerNet | DeiT-S        | 512x512  | 80000  | |  | 42.96 |  |  |  |
| UPerNet | DeiT-S        | 512x512  | 160000 | |  | 42.87 |  |  |  |
| UPerNet | DeiT-S + neck | 512x512  | 160000 | |  | 43.82 |  |  |  |
| UPerNet | DeiT-S + norm | 512x512  | 160000 | |  | 43.52 |  |  |  |
| UPerNet | DeiT-B        | 512x512  | 80000  | |  | 45.24 |  |  |  |
| UPerNet | DeiT-B        | 512x512  | 160000 | |  | 45.36 |  |  |  |
| UPerNet | DeiT-B + neck | 512x512  | 160000 | |  | 45.46 |  |  |  |
| UPerNet | DeiT-B + norm | 512x512  | 160000 | |  | 45.37 |  |  |  |
