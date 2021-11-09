# ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/Eromera/erfnet_pytorch">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.20.0/mmseg/models/backbones/erfnet.py#L321">Code Snippet</a>

<details>
<summary align="right"><a href="http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf">ERFNet (T-ITS)</a></summary>

```latex
@article{romera2017erfnet,
  title={Erfnet: Efficient residual factorized convnet for real-time semantic segmentation},
  author={Romera, Eduardo and Alvarez, Jos{\'e} M and Bergasa, Luis M and Arroyo, Roberto},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={19},
  number={1},
  pages={263--272},
  year={2017},
  publisher={IEEE}
}
```

</details>

## Results and models

### Cityscapes

| Method    | Backbone  | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                  | download                                                                                                                                                                                                                                                       |
| --------- | --------- | --------- | ------: | -------- | -------------- | ----: | ------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FCN | ERFNet | 512x1024  | 160000 | 16.40 | 2.16 | 71.4 | 72.96 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/erfnet/fcn_erfnet_4x4_512x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/erfnet/fcn_erfnet_4x4_512x1024_160k_cityscapes/fcn_erfnet_4x4_512x1024_160k_cityscapes_20211103_011334-8f691334.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/erfnet/fcn_erfnet_4x4_512x1024_160k_cityscapes/fcn_erfnet_4x4_512x1024_160k_cityscapes_20211103_011334.log.json) |
Note:

- Last deconvolution layer in the original paper is replaced by a naive `FCN` decoder head and a bilinear upsampling layer.
