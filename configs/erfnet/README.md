# ERFNet

[ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/Eromera/erfnet_pytorch">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.20.0/mmseg/models/backbones/erfnet.py#L321">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Semantic segmentation is a challenging task that addresses most of the perception needs of intelligent vehicles (IVs) in an unified way. Deep neural networks excel at this task, as they can be trained end-to-end to accurately classify multiple object categories in an image at pixel level. However, a good tradeoff between high quality and computational resources is yet not present in the state-of-the-art semantic segmentation approaches, limiting their application in real vehicles. In this paper, we propose a deep architecture that is able to run in real time while providing accurate semantic segmentation. The core of our architecture is a novel layer that uses residual connections and factorized convolutions in order to remain efficient while retaining remarkable accuracy. Our approach is able to run at over 83 FPS in a single Titan X, and 7 FPS in a Jetson TX1 (embedded device). A comprehensive set of experiments on the publicly available Cityscapes data set demonstrates that our system achieves an accuracy that is similar to the state of the art, while being orders of magnitude faster to compute than other architectures that achieve top precision. The resulting tradeoff makes our model an ideal approach for scene understanding in IV applications. The code is publicly available at: https://github.com/Eromera/erfnet.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/143479729-ea7951f6-1a3c-47d6-aaee-62c5759c0638.png" width="60%"/>
</div>

## Citation

```bibtex
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

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                       | download                                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ERFNet | ERFNet   | 512x1024  |  160000 | 6.04     | 15.26          | 71.08 | 72.6          | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/erfnet/erfnet_fcn_4x4_512x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/erfnet/erfnet_fcn_4x4_512x1024_160k_cityscapes/erfnet_fcn_4x4_512x1024_160k_cityscapes_20211126_082056-03d333ed.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/erfnet/erfnet_fcn_4x4_512x1024_160k_cityscapes/erfnet_fcn_4x4_512x1024_160k_cityscapes_20211126_082056.log.json) |

Note:

- The model is trained from scratch.

- Last deconvolution layer in the [original paper](https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py#L123) is replaced by a naive `FCNHead` decoder head and a bilinear upsampling layer, found more effective and efficient.
