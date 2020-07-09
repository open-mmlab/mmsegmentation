<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
</div>

## Introduction

MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.
It is a part of the OpenMMLab project.

The master branch works with **PyTorch 1.3 to 1.5**.

![demo image](resources/seg_demo.gif)

### Major features

- **Unified Benchmark**

  We provide a unified benchmark toolbox for various semantic segmentation methods.

- **Modular Design**

  We decompose the semantic segmentation framework into different components and one can easily construct a customized semantic segmentation framework by combining different modules.

- **Support of multiple methods out of box**

  The toolbox directly supports popular and contemporary semantic segmentation frameworks, *e.g.* PSPNet, DeepLabV3, PSANet, DeepLabV3+, etc.

- **High efficiency**

  The training speed is faster than or comparable to other codebases.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:
- [x] ResNet
- [x] ResNeXt
- [x] HRNet

Supported methods:
- [x] [FCN](configs/fcn)
- [x] [PSPNet](configs/pspnet)
- [x] [DeepLabV3](configs/deeplabv3)
- [x] [PSANet](configs/psanet)
- [x] [DeepLabV3+](configs/deeplabv3plus)
- [x] [UPerNet](configs/upernet)
- [x] [NonLocal Net](configs/nonlocal_net)
- [x] [CCNet](configs/ccnet)
- [x] [DANet](configs/danet)
- [x] [GCNet](configs/gcnet)
- [x] [ANN](configs/ann)
- [x] [OCRNet](configs/ocrnet)

## Installation

Please refer to [INSTALL.md](docs/install.md) for installation and dataset preparation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMSegmentation.
There are also tutorials for [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).

## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

Many thanks to Ruobing Han ([@drcut](https://github.com/drcut)), Xiaoming Ma([@aishangmaxiaoming](https://github.com/aishangmaxiaoming)), Shiguang Wang ([@sunnyxiaohu](https://github.com/sunnyxiaohu)) for deployment support.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@misc{mmseg2020,
  author={Xu, Jiarui and Chen, Kai and Lin, Dahua},
  title={{MMSegmenation}},
  howpublished={\url{https://github.com/open-mmlab/mmsegmentation}},
  year={2020}
}
```

## Contact

This repo is currently maintained by Jiarui Xu ([@xvjiarui](https://github.com/xvjiarui)), Kai Chen ([@hellock](http://github.com/hellock)).
