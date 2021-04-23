# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## Introduction

[ALGORITHM]

```latex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## Results and models

### ADE20K

| Backbone | Method | Crop Size | Lr Schd | mIoU | mIoU (ms+flip) | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | UPerNet | 512x512 | 160K | 44.51 | 45.81 | 60M | 945G | [config](configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.log.json)/[baidu](https://pan.baidu.com/s/1dq0DdS17dFcmAzHlM_1rgw) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth)/[baidu](https://pan.baidu.com/s/17VmmppX-PUKuek9T5H3Iqw) |
| Swin-S | UperNet | 512x512 | 160K | 47.64 | 49.47 | 81M | 1038G | [config](configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_small_patch4_window7_512x512.log.json)/[baidu](https://pan.baidu.com/s/1ko3SVKPzH9x5B7SWCFxlig) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_small_patch4_window7_512x512.pth)/[baidu](https://pan.baidu.com/s/184em63etTMsf0cR_NX9zNg) |
| Swin-B | UperNet | 512x512 | 160K | 48.13 | 49.72 | 121M | 1188G | [config](configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.log.json)/[baidu](https://pan.baidu.com/s/1YlXXiB3GwUKhHobUajlIaQ) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.pth)/[baidu](https://pan.baidu.com/s/12B2dY_niMirwtu64_9AMbg) |

**Notes**: 

- **Pre-trained models can be downloaded from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer)**.
- Access code for `baidu` is `swin`.