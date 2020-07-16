# Mixed Precision Training

## Introduction
```
@article{micikevicius2017mixed,
  title={Mixed precision training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and others},
  journal={arXiv preprint arXiv:1710.03740},
  year={2017}
}
```

## Results and models

### Cityscapes
| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                        download                                                                                                                                                                                        |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCN    | R-101-D8 | 512x1024  |   80000 | 5.50        | -              | - |         - | [model]() &#124; [log]() |
| PSPNet    | R-101-D8 | 512x1024  |   80000 | 5.47        | -              | - |         - | [model]() &#124; [log]() |
| DeepLabV3    | R-101-D8 | 512x1024  |   80000 | 5.91        | -              | - |         - | [model]() &#124; [log]() |
| DeepLabV3+    | R-101-D8 | 512x1024  |   80000 | 6.46        | -              | - |         - | [model]() &#124; [log]() |
