<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmsegmentation)](https://pypi.org/project/mmsegmentation/)
[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/zh_CN/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)

文档: <https://mmsegmentation.readthedocs.io/zh_CN/latest>

[English](README.md) | 简体中文

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## 简介

MMSegmentation 是一个基于 PyTorch 的语义分割开源工具箱。它是 OpenMMLab 项目的一部分。

[main](https://github.com/open-mmlab/mmsegmentation/tree/main) 分支代码目前支持 PyTorch 1.6 以上的版本。

### 🎉 MMSegmentation v1.0.0 简介 🎉

我们非常高兴地宣布 MMSegmentation 最新版本的正式发布！在这个新版本中，主要分支是 [main](https://github.com/open-mmlab/mmsegmentation/tree/main) 分支，开发分支是 [dev-1.x](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x)。而之前版本的稳定分支保留为 [0.x](https://github.com/open-mmlab/mmsegmentation/tree/0.x) 分支。请注意，[master](https://github.com/open-mmlab/mmsegmentation/tree/master) 分支将只在有限的时间内维护，然后将被删除。我们鼓励您在使用过程中注意分支选择和更新。感谢您一如既往的支持和热情，让我们共同努力，使 MMSegmentation 变得更加健壮和强大！💪

MMSegmentation v1.x 在 0.x 版本的基础上有了显著的提升，提供了更加灵活和功能丰富的体验。为了更好使用 v1.x 中的新功能，我们诚挚邀请您查阅我们详细的 [📚 迁移指南](https://mmsegmentation.readthedocs.io/zh_CN/latest/migration/interface.html)，以帮助您无缝地过渡您的项目。您的支持对我们来说非常宝贵，我们热切期待您的反馈！

![示例图片](resources/seg_demo.gif)

### 主要特性

- **统一的基准平台**

  我们将各种各样的语义分割算法集成到了一个统一的工具箱，进行基准测试。

- **模块化设计**

  MMSegmentation 将分割框架解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的分割模型。

- **丰富的即插即用的算法和模型**

  MMSegmentation 支持了众多主流的和最新的检测算法，例如 PSPNet，DeepLabV3，PSANet，DeepLabV3+ 等.

- **速度快**

  训练速度比其他语义分割代码库更快或者相当。

## 更新日志

最新版本 v1.0.0 在 2023.04.06 发布。
如果想了解更多版本更新细节和历史信息，请阅读[更新日志](docs/en/notes/changelog.md)。

## 安装

请参考[快速入门文档](docs/zh_cn/get_started.md#installation)进行安装，参考[数据集准备](docs/zh_cn/user_guides/2_dataset_prepare.md)处理数据。

## 快速入门

请参考[概述](docs/zh_cn/overview.md)对 MMSegmetation 进行初步了解

请参考[用户指南](https://mmsegmentation.readthedocs.io/zh_CN/latest/user_guides/index.html)了解 mmseg 的基本使用，以及[进阶指南](https://mmsegmentation.readthedocs.io/zh_CN/latest/advanced_guides/index.html)深入了解 mmseg 设计和代码实现。

同时，我们提供了 Colab 教程。你可以在[这里](demo/MMSegmentation_Tutorial.ipynb)浏览教程，或者直接在 Colab 上[运行](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/main/demo/MMSegmentation_Tutorial.ipynb)。

若需要将 0.x 版本的代码迁移至新版，请参考[迁移文档](docs/zh_cn/migration)。

## 教程文档

<details>
<summary>开启 MMSeg 之旅</summary>

- [MMSeg 概述](docs/zh_cn/overview.md)
- [安装和验证](docs/zh_cn/get_started.md)
- [常见问题解答](docs/zh_cn/notes/faq.md)

</details>

<details>
<summary>MMSeg 快速入门教程</summary>

- [教程1：了解配置文件](docs/zh_cn/user_guides/1_config.md)
- [教程2：准备数据集](docs/zh_cn/user_guides/2_dataset_prepare.md)
- [教程3：使用预训练模型推理](docs/zh_cn/user_guides/3_inference.md)
- [教程4：使用现有模型进行训练和测试](docs/zh_cn/user_guides/4_train_test.md)
- [教程5：模型部署](docs/zh_cn/user_guides/5_deployment.md)
- [常用工具](docs/zh_cn/user_guides/useful_tools.md)
- [特征图可视化](docs/zh_cn/user_guides/visualization_feature_map.md)
- [可视化](docs/zh_cn/user_guides/visualization.md)

</details>

<details>
<summary>MMSeg 细节介绍</summary>

- [MMSeg 数据集介绍](docs/zh_cn/advanced_guides/datasets.md)
- [MMSeg 模型介绍](docs/zh_cn/advanced_guides/models.md)
- [MMSeg 数据结构介绍](docs/zh_cn/advanced_guides/structures.md)
- [MMSeg 数据增强介绍](docs/zh_cn/advanced_guides/transforms.md)
- [MMSeg 数据流介绍](docs/zh_cn/advanced_guides/data_flow.md)
- [MMSeg 训练引擎介绍](docs/zh_cn/advanced_guides/engine.md)
- [MMSeg 模型评测介绍](docs/zh_cn/advanced_guides/evaluation.md)

</details>

<details>
<summary>MMSeg 开发教程</summary>

- [新增自定义数据集](docs/zh_cn/advanced_guides/add_datasets.md)
- [新增评测指标](docs/zh_cn/advanced_guides/add_metrics.md)
- [新增自定义模型](docs/zh_cn/advanced_guides/add_models.md)
- [新增自定义数据增强](docs/zh_cn/advanced_guides/add_transforms.md)
- [自定义运行设定](docs/zh_cn/advanced_guides/customize_runtime.md)
- [训练技巧](docs/zh_cn/advanced_guides/training_tricks.md)
- [如何给 MMSeg贡献代码](.github/CONTRIBUTING.md)
- [在 projects 给 MMSeg 贡献一个标准数据集](docs/zh_cn/advanced_guides/contribute_dataset.md)
- [NPU (华为 昇腾)](docs/zh_cn/device/npu.md)
- [0.x → 1.x 迁移文档](docs/zh_cn/migration/interface.md)，[0.x → 1.x 库变更文档](docs/zh_cn/migration/package.md)

</details>

## 基准测试和模型库

测试结果和模型可以在[模型库](docs/zh_cn/model_zoo.md)中找到。

<details open>
<summary>已支持的骨干网络：</summary>

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] [HRNet (CVPR'2019)](configs/hrnet)
- [x] [ResNeSt (ArXiv'2020)](configs/resnest)
- [x] [MobileNetV2 (CVPR'2018)](configs/mobilenet_v2)
- [x] [MobileNetV3 (ICCV'2019)](configs/mobilenet_v3)
- [x] [Vision Transformer (ICLR'2021)](configs/vit)
- [x] [Swin Transformer (ICCV'2021)](configs/swin)
- [x] [Twins (NeurIPS'2021)](configs/twins)
- [x] [BEiT (ICLR'2022)](configs/beit)
- [x] [ConvNeXt (CVPR'2022)](configs/convnext)
- [x] [MAE (CVPR'2022)](configs/mae)
- [x] [PoolFormer (CVPR'2022)](configs/poolformer)
- [x] [SegNeXt (NeurIPS'2022)](configs/segnext)

</details>

<details open>
<summary>已支持的算法：</summary>

- [x] [FCN (CVPR'2015/TPAMI'2017)](configs/fcn)
- [x] [ERFNet (T-ITS'2017)](configs/erfnet)
- [x] [UNet (MICCAI'2016/Nat. Methods'2019)](configs/unet)
- [x] [PSPNet (CVPR'2017)](configs/pspnet)
- [x] [DeepLabV3 (ArXiv'2017)](configs/deeplabv3)
- [x] [BiSeNetV1 (ECCV'2018)](configs/bisenetv1)
- [x] [PSANet (ECCV'2018)](configs/psanet)
- [x] [DeepLabV3+ (CVPR'2018)](configs/deeplabv3plus)
- [x] [UPerNet (ECCV'2018)](configs/upernet)
- [x] [ICNet (ECCV'2018)](configs/icnet)
- [x] [NonLocal Net (CVPR'2018)](configs/nonlocal_net)
- [x] [EncNet (CVPR'2018)](configs/encnet)
- [x] [Semantic FPN (CVPR'2019)](configs/sem_fpn)
- [x] [DANet (CVPR'2019)](configs/danet)
- [x] [APCNet (CVPR'2019)](configs/apcnet)
- [x] [EMANet (ICCV'2019)](configs/emanet)
- [x] [CCNet (ICCV'2019)](configs/ccnet)
- [x] [DMNet (ICCV'2019)](configs/dmnet)
- [x] [ANN (ICCV'2019)](configs/ann)
- [x] [GCNet (ICCVW'2019/TPAMI'2020)](configs/gcnet)
- [x] [FastFCN (ArXiv'2019)](configs/fastfcn)
- [x] [Fast-SCNN (ArXiv'2019)](configs/fastscnn)
- [x] [ISANet (ArXiv'2019/IJCV'2021)](configs/isanet)
- [x] [OCRNet (ECCV'2020)](configs/ocrnet)
- [x] [DNLNet (ECCV'2020)](configs/dnlnet)
- [x] [PointRend (CVPR'2020)](configs/point_rend)
- [x] [CGNet (TIP'2020)](configs/cgnet)
- [x] [BiSeNetV2 (IJCV'2021)](configs/bisenetv2)
- [x] [STDC (CVPR'2021)](configs/stdc)
- [x] [SETR (CVPR'2021)](configs/setr)
- [x] [DPT (ArXiv'2021)](configs/dpt)
- [x] [Segmenter (ICCV'2021)](configs/segmenter)
- [x] [SegFormer (NeurIPS'2021)](configs/segformer)
- [x] [K-Net (NeurIPS'2021)](configs/knet)
- [x] [MaskFormer (NeurIPS'2021)](configs/maskformer)
- [x] [Mask2Former (CVPR'2022)](configs/mask2former)
- [x] [PIDNet (ArXiv'2022)](configs/pidnet)
- [x] [DDRNet (T-ITS'2022)](configs/ddrnet)

</details>

<details open>
<summary>已支持的数据集：</summary>

- [x] [Cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#cityscapes)
- [x] [PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#pascal-voc)
- [x] [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#ade20k)
- [x] [Pascal Context](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#pascal-context)
- [x] [COCO-Stuff 10k](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#coco-stuff-10k)
- [x] [COCO-Stuff 164k](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#coco-stuff-164k)
- [x] [CHASE_DB1](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#chase-db1)
- [x] [DRIVE](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#drive)
- [x] [HRF](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#hrf)
- [x] [STARE](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#stare)
- [x] [Dark Zurich](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#dark-zurich)
- [x] [Nighttime Driving](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#nighttime-driving)
- [x] [LoveDA](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#loveda)
- [x] [Potsdam](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#isprs-potsdam)
- [x] [Vaihingen](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#isprs-vaihingen)
- [x] [iSAID](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/dataset_prepare.md#isaid)
- [x] [Mapillary Vistas](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#mapillary-vistas-datasets)

</details>

如果遇到问题，请参考 [常见问题解答](docs/zh_cn/notes/faq.md)。

## 社区项目

[这里](projects/README.md)有一些由社区用户支持和维护的基于 MMSegmentation 的 SOTA 模型和解决方案的实现。这些项目展示了基于 MMSegmentation 的研究和产品开发的最佳实践。
我们欢迎并感谢对 OpenMMLab 生态系统的所有贡献。

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMSegmentation 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

MMSegmentation 是一个由来自不同高校和企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。我们希望这个工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现已有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 引用

如果你觉得本项目对你的研究工作有所帮助，请参考如下 bibtex 引用 MMSegmentation。

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## OpenMMLab 的其他项目

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab 深度学习预训练工具箱
- [MMagic](https://github.com/open-mmlab/mmagic): OpenMMLab 新一代人工智能内容生成（AIGC）工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MIM](https://github.com/open-mmlab/mim): OpenMMLab 项目、算法、模型的统一入口
- [Playground](https://github.com/open-mmlab/playground): 收集和展示 OpenMMLab 相关的前沿、有趣的社区项目

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 [OpenMMLab 团队](https://jq.qq.com/?_wv=1027&k=aCvMxdr3) 以及 [MMSegmentation](https://jq.qq.com/?_wv=1027&k=ukevz6Ie) 的 QQ 群。

<div align="center">
 <img src="docs/zh_cn/imgs/zhihu_qrcode.jpg" height="400" />  <img src="docs/zh_cn/imgs/qq_group_qrcode.jpg" height="400" />  <img src="docs/zh_cn/imgs/seggroup_qrcode.jpg" height="400" />
 </div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
