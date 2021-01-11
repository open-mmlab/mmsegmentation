# U-Net: Convolutional Networks for Biomedical Image Segmentation

## Introduction

[ALGORITHM]

```latex
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

## Results and models

| Backbone | Head | Dataset | Image Size | Crop Size | Stride | Lr schd | Mem (GB) | Inf time (fps) | Dice  |                                                                                                                                                                                         download                                                                                                                                                                                         |
|--------|----------|----------|----------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| UNet-S5-D16 | FCN  | DRIVE   |   584x565 |      64x64 |          42x42 | 40000 |         0.680 |  - | 78.67 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_64x64_40k_drive/unet_s5-d16_64x64_40k_drive_20201223_191051-9cd163b8.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_64x64_40k_drive/unet_s5-d16_64x64_40k_drive-20201223_191051.log.json)         |
| UNet-S5-D16 | FCN  | STARE   |   605x700 |      128x128 |          85x85 | 40000 |         0.968 |  - | 81.02 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_128x128_40k_stare/unet_s5-d16_128x128_40k_stare_20201223_191051-e5439846.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_128x128_40k_stare/unet_s5-d16_128x128_40k_stare-20201223_191051.log.json)         |
| UNet-S5-D16 | FCN  | CHASE_DB1   |   960x999 |      128x128 |          85x85 | 40000 |         0.968 |  - | 80.24 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_128x128_40k_chase_db1/unet_s5-d16_128x128_40k_chase_db1_20201223_191051-8b16ca0b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_128x128_40k_chase_db1/unet_s5-d16_128x128_40k_chase_db1-20201223_191051.log.json)         |
| UNet-S5-D16 | FCN  | HRF   |   2336x3504 |      256x256 |          170x170 | 40000 |         2.525 |  - | 79.45 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_256x256_40k_hrf/unet_s5-d16_256x256_40k_hrf_20201223_173724-d89cf1ed.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_256x256_40k_hrf/unet_s5-d16_256x256_40k_hrf-20201223_173724.log.json)         |
