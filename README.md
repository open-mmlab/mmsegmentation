<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmsegmentation)](https://pypi.org/project/mmsegmentation/)
[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_demo.svg)](https://openxlab.org.cn/apps?search=mmseg)

Documentation: <https://mmsegmentation.readthedocs.io/en/latest/>

English | [简体中文](README_zh-CN.md)

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

## Introduction

MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.
It is a part of the OpenMMLab project.

The [main](https://github.com/open-mmlab/mmsegmentation/tree/main) branch works with PyTorch 1.6+.

### 🎉 Introducing MMSegmentation v1.0.0 🎉

We are thrilled to announce the official release of MMSegmentation's latest version! For this new release, the [main](https://github.com/open-mmlab/mmsegmentation/tree/main) branch serves as the primary branch, while the development branch is [dev-1.x](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x). The stable branch for the previous release remains as the [0.x](https://github.com/open-mmlab/mmsegmentation/tree/0.x) branch. Please note that the [master](https://github.com/open-mmlab/mmsegmentation/tree/master) branch will only be maintained for a limited time before being removed. We encourage you to be mindful of branch selection and updates during use. Thank you for your unwavering support and enthusiasm, and let's work together to make MMSegmentation even more robust and powerful! 💪

MMSegmentation v1.x brings remarkable improvements over the 0.x release, offering a more flexible and feature-packed experience. To utilize the new features in v1.x, we kindly invite you to consult our detailed [📚 migration guide](https://mmsegmentation.readthedocs.io/en/latest/migration/interface.html), which will help you seamlessly transition your projects. Your support is invaluable, and we eagerly await your feedback!

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

## What's New

v1.2.0 was released on 10/12/2023, from 1.1.0 to 1.2.0, we have added or updated the following features:

### Highlights

- Support for the open-vocabulary semantic segmentation algorithm [SAN](configs/san/README.md)

- Support monocular depth estimation task, please refer to [VPD](configs/vpd/README.md) and [Adabins](projects/Adabins/README.md) for more details.

  ![depth estimation](https://github.com/open-mmlab/mmsegmentation/assets/15952744/07afd0e9-8ace-4a00-aa1e-5bf0ca92dcbc)

- Add new projects: open-vocabulary semantic segmentation algorithm [CAT-Seg](projects/CAT-Seg/README.md), real-time semantic segmentation algofithm [PP-MobileSeg](projects/pp_mobileseg/README.md)

## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

Please see [Overview](docs/en/overview.md) for the general introduction of MMSegmentation.

Please see [user guides](https://mmsegmentation.readthedocs.io/en/latest/user_guides/index.html#) for the basic usage of MMSegmentation.
There are also [advanced tutorials](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/index.html) for in-depth understanding of mmseg design and implementation .

A Colab tutorial is also provided. You may preview the notebook [here](demo/MMSegmentation_Tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/main/demo/MMSegmentation_Tutorial.ipynb) on Colab.

To migrate from MMSegmentation 0.x, please refer to [migration](docs/en/migration).

## Tutorial

<div align="center">
  <b>MMSegmentation Tutorials</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Get Started</b>
      </td>
      <td>
        <b>MMSeg Basic Tutorial</b>
      </td>
      <td>
        <b>MMSeg Detail Tutorial</b>
      </td>
      <td>
        <b>MMSeg Development Tutorial</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><a href="docs/en/overview.md">MMSeg overview</a></li>
          <li><a href="docs/en/get_started.md">MMSeg Installation</a></li>
          <li><a href="docs/en/notes/faq.md">FAQ</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="docs/en/user_guides/1_config.md">Tutorial 1: Learn about Configs</a></li>
          <li><a href="docs/en/user_guides/2_dataset_prepare.md">Tutorial 2: Prepare datasets</a></li>
          <li><a href="docs/en/user_guides/3_inference.md">Tutorial 3: Inference with existing models</a></li>
          <li><a href="docs/en/user_guides/4_train_test.md">Tutorial 4: Train and test with existing models</a></li>
          <li><a href="docs/en/user_guides/5_deployment.md">Tutorial 5: Model deployment</a></li>
          <li><a href="docs/zh_cn/user_guides/deploy_jetson.md">Deploy mmsegmentation on Jetson platform</a></li>
          <li><a href="docs/en/user_guides/useful_tools.md">Useful Tools</a></li>
          <li><a href="docs/en/user_guides/visualization_feature_map.md">Feature Map Visualization</a></li>
          <li><a href="docs/en/user_guides/visualization.md">Visualization</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="docs/en/advanced_guides/datasets.md">MMSeg Dataset</a></li>
          <li><a href="docs/en/advanced_guides/models.md">MMSeg Models</a></li>
          <li><a href="docs/en/advanced_guides/structures.md">MMSeg Dataset Structures</a></li>
          <li><a href="docs/en/advanced_guides/transforms.md">MMSeg Data Transforms</a></li>
          <li><a href="docs/en/advanced_guides/data_flow.md">MMSeg Dataflow</a></li>
          <li><a href="docs/en/advanced_guides/engine.md">MMSeg Training Engine</a></li>
          <li><a href="docs/en/advanced_guides/evaluation.md">MMSeg Evaluation</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="docs/en/advanced_guides/add_datasets.md">Add New Datasets</a></li>
          <li><a href="docs/en/advanced_guides/add_metrics.md">Add New Metrics</a></li>
          <li><a href="docs/en/advanced_guides/add_models.md">Add New Modules</a></li>
          <li><a href="docs/en/advanced_guides/add_transforms.md">Add New Data Transforms</a></li>
          <li><a href="docs/en/advanced_guides/customize_runtime.md">Customize Runtime Settings</a></li>
          <li><a href="docs/en/advanced_guides/training_tricks.md">Training Tricks</a></li>
          <li><a href=".github/CONTRIBUTING.md">Contribute code to MMSeg</a></li>
          <li><a href="docs/zh_cn/advanced_guides/contribute_dataset.md">Contribute a standard dataset in projects</a></li>
          <li><a href="docs/en/device/npu.md">NPU (HUAWEI Ascend)</a></li>
          <li><a href="docs/en/migration/interface.md">0.x → 1.x migration</a></li>
          <li><a href="docs/en/migration/package.md">0.x → 1.x package</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

<div align="center">
  <b>Overview</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Supported backbones</b>
      </td>
      <td>
        <b>Supported methods</b>
      </td>
      <td>
        <b>Supported Head</b>
      </td>
      <td>
        <b>Supported datasets</b>
      </td>
      <td>
        <b>Other</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <li><a href="mmseg/models/backbones/resnet.py">ResNet(CVPR'2016)</a></li>
        <li><a href="mmseg/models/backbones/resnext.py">ResNeXt (CVPR'2017)</a></li>
        <li><a href="configs/hrnet">HRNet (CVPR'2019)</a></li>
        <li><a href="configs/resnest">ResNeSt (ArXiv'2020)</a></li>
        <li><a href="configs/mobilenet_v2">MobileNetV2 (CVPR'2018)</a></li>
        <li><a href="configs/mobilenet_v3">MobileNetV3 (ICCV'2019)</a></li>
        <li><a href="configs/vit">Vision Transformer (ICLR'2021)</a></li>
        <li><a href="configs/swin">Swin Transformer (ICCV'2021)</a></li>
        <li><a href="configs/twins">Twins (NeurIPS'2021)</a></li>
        <li><a href="configs/beit">BEiT (ICLR'2022)</a></li>
        <li><a href="configs/convnext">ConvNeXt (CVPR'2022)</a></li>
        <li><a href="configs/mae">MAE (CVPR'2022)</a></li>
        <li><a href="configs/poolformer">PoolFormer (CVPR'2022)</a></li>
        <li><a href="configs/segnext">SegNeXt (NeurIPS'2022)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/san/">SAN (CVPR'2023)</a></li>
          <li><a href="configs/vpd">VPD (ICCV'2023)</a></li>
          <li><a href="configs/ddrnet">DDRNet (T-ITS'2022)</a></li>
          <li><a href="configs/pidnet">PIDNet (ArXiv'2022)</a></li>
          <li><a href="configs/mask2former">Mask2Former (CVPR'2022)</a></li>
          <li><a href="configs/maskformer">MaskFormer (NeurIPS'2021)</a></li>
          <li><a href="configs/knet">K-Net (NeurIPS'2021)</a></li>
          <li><a href="configs/segformer">SegFormer (NeurIPS'2021)</a></li>
          <li><a href="configs/segmenter">Segmenter (ICCV'2021)</a></li>
          <li><a href="configs/dpt">DPT (ArXiv'2021)</a></li>
          <li><a href="configs/setr">SETR (CVPR'2021)</a></li>
          <li><a href="configs/stdc">STDC (CVPR'2021)</a></li>
          <li><a href="configs/bisenetv2">BiSeNetV2 (IJCV'2021)</a></li>
          <li><a href="configs/cgnet">CGNet (TIP'2020)</a></li>
          <li><a href="configs/point_rend">PointRend (CVPR'2020)</a></li>
          <li><a href="configs/dnlnet">DNLNet (ECCV'2020)</a></li>
          <li><a href="configs/ocrnet">OCRNet (ECCV'2020)</a></li>
          <li><a href="configs/isanet">ISANet (ArXiv'2019/IJCV'2021)</a></li>
          <li><a href="configs/fastscnn">Fast-SCNN (ArXiv'2019)</a></li>
          <li><a href="configs/fastfcn">FastFCN (ArXiv'2019)</a></li>
          <li><a href="configs/gcnet">GCNet (ICCVW'2019/TPAMI'2020)</a></li>
          <li><a href="configs/ann">ANN (ICCV'2019)</a></li>
          <li><a href="configs/emanet">EMANet (ICCV'2019)</a></li>
          <li><a href="configs/ccnet">CCNet (ICCV'2019)</a></li>
          <li><a href="configs/dmnet">DMNet (ICCV'2019)</a></li>
          <li><a href="configs/sem_fpn">Semantic FPN (CVPR'2019)</a></li>
          <li><a href="configs/danet">DANet (CVPR'2019)</a></li>
          <li><a href="configs/apcnet">APCNet (CVPR'2019)</a></li>
          <li><a href="configs/nonlocal_net">NonLocal Net (CVPR'2018)</a></li>
          <li><a href="configs/encnet">EncNet (CVPR'2018)</a></li>
          <li><a href="configs/deeplabv3plus">DeepLabV3+ (CVPR'2018)</a></li>
          <li><a href="configs/upernet">UPerNet (ECCV'2018)</a></li>
          <li><a href="configs/icnet">ICNet (ECCV'2018)</a></li>
          <li><a href="configs/psanet">PSANet (ECCV'2018)</a></li>
          <li><a href="configs/bisenetv1">BiSeNetV1 (ECCV'2018)</a></li>
          <li><a href="configs/deeplabv3">DeepLabV3 (ArXiv'2017)</a></li>
          <li><a href="configs/pspnet">PSPNet (CVPR'2017)</a></li>
          <li><a href="configs/erfnet">ERFNet (T-ITS'2017)</a></li>
          <li><a href="configs/unet">UNet (MICCAI'2016/Nat. Methods'2019)</a></li>
          <li><a href="configs/fcn">FCN (CVPR'2015/TPAMI'2017)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="mmseg/models/decode_heads/ann_head.py">ANN_Head</li>
          <li><a href="mmseg/models/decode_heads/apc_head.py">APC_Head</li>
          <li><a href="mmseg/models/decode_heads/aspp_head.py">ASPP_Head</li>
          <li><a href="mmseg/models/decode_heads/cc_head.py">CC_Head</li>
          <li><a href="mmseg/models/decode_heads/da_head.py">DA_Head</li>
          <li><a href="mmseg/models/decode_heads/ddr_head.py">DDR_Head</li>
          <li><a href="mmseg/models/decode_heads/dm_head.py">DM_Head</li>
          <li><a href="mmseg/models/decode_heads/dnl_head.py">DNL_Head</li>
          <li><a href="mmseg/models/decode_heads/dpt_head.py">DPT_HEAD</li>
          <li><a href="mmseg/models/decode_heads/ema_head.py">EMA_Head</li>
          <li><a href="mmseg/models/decode_heads/enc_head.py">ENC_Head</li>
          <li><a href="mmseg/models/decode_heads/fcn_head.py">FCN_Head</li>
          <li><a href="mmseg/models/decode_heads/fpn_head.py">FPN_Head</li>
          <li><a href="mmseg/models/decode_heads/gc_head.py">GC_Head</li>
          <li><a href="mmseg/models/decode_heads/ham_head.py">LightHam_Head</li>
          <li><a href="mmseg/models/decode_heads/isa_head.py">ISA_Head</li>
          <li><a href="mmseg/models/decode_heads/knet_head.py">Knet_Head</li>
          <li><a href="mmseg/models/decode_heads/lraspp_head.py">LRASPP_Head</li>
          <li><a href="mmseg/models/decode_heads/mask2former_head.py">mask2former_Head</li>
          <li><a href="mmseg/models/decode_heads/maskformer_head.py">maskformer_Head</li>
          <li><a href="mmseg/models/decode_heads/nl_head.py">NL_Head</li>
          <li><a href="mmseg/models/decode_heads/ocr_head.py">OCR_Head</li>
          <li><a href="mmseg/models/decode_heads/pid_head.py">PID_Head</li>
          <li><a href="mmseg/models/decode_heads/point_head.py">point_Head</li>
          <li><a href="mmseg/models/decode_heads/psa_head.py">PSA_Head</li>
          <li><a href="mmseg/models/decode_heads/psp_head.py">PSP_Head</li>
          <li><a href="mmseg/models/decode_heads/san_head.py">SAN_Head</li>
          <li><a href="mmseg/models/decode_heads/segformer_head.py">segformer_Head</li>
          <li><a href="mmseg/models/decode_heads/segmenter_mask_head.py">segmenter_mask_Head</li>
          <li><a href="mmseg/models/decode_heads/sep_aspp_head.py">SepASPP_Head</li>
          <li><a href="mmseg/models/decode_heads/sep_fcn_head.py">SepFCN_Head</li>
          <li><a href="mmseg/models/decode_heads/setr_mla_head.py">SETRMLAHead_Head</li>
          <li><a href="mmseg/models/decode_heads/setr_up_head.py">SETRUP_Head</li>
          <li><a href="mmseg/models/decode_heads/stdc_head.py">STDC_Head</li>
          <li><a href="mmseg/models/decode_heads/uper_head.py">Uper_Head</li>
          <li><a href="mmseg/models/decode_heads/vpd_depth_head.py">VPDDepth_Head</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#cityscapes">Cityscapes</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-voc">PASCAL VOC</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#ade20k">ADE20K</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context">Pascal Context</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#coco-stuff-10k">COCO-Stuff 10k</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#coco-stuff-164k">COCO-Stuff 164k</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#chase-db1">CHASE_DB1</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#drive">DRIVE</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#hrf">HRF</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#stare">STARE</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#dark-zurich">Dark Zurich</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#nighttime-driving">Nighttime Driving</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#loveda">LoveDA</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isprs-potsdam">Potsdam</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isprs-vaihingen">Vaihingen</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isaid">iSAID</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#mapillary-vistas-datasets">Mapillary Vistas</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#levir-cd">LEVIR-CD</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#bdd100K">BDD100K</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#nyu">NYU</a></li>
          <li><a href="https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#hsi-drive-2.0">HSIDrive20</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><b>Supported loss</b></li>
        <ul>
          <li><a href="mmseg/models/losses/boundary_loss.py">boundary_loss</a></li>
          <li><a href="mmseg/models/losses/cross_entropy_loss.py">cross_entropy_loss</a></li>
          <li><a href="mmseg/models/losses/dice_loss.py">dice_loss</a></li>
          <li><a href="mmseg/models/losses/focal_loss.py">focal_loss</a></li>
          <li><a href="mmseg/models/losses/huasdorff_distance_loss.py">huasdorff_distance_loss</a></li>
          <li><a href="mmseg/models/losses/kldiv_loss.py">kldiv_loss</a></li>
          <li><a href="mmseg/models/losses/lovasz_loss.py">lovasz_loss</a></li>
          <li><a href="mmseg/models/losses/ohem_cross_entropy_loss.py">ohem_cross_entropy_loss</a></li>
          <li><a href="mmseg/models/losses/silog_loss.py">silog_loss</a></li>
          <li><a href="mmseg/models/losses/tversky_loss.py">tversky_loss</a></li>
        </ul>
        </ul>
      </td>
  </tbody>
</table>

Please refer to [FAQ](docs/en/notes/faq.md) for frequently asked questions.

## Projects

[Here](projects/README.md) are some implementations of SOTA models and solutions built on MMSegmentation, which are supported and maintained by community users. These projects demonstrate the best practices based on MMSegmentation for research and product development. We welcome and appreciate all the contributions to OpenMMLab ecosystem.

## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## OpenMMLab Family

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab pre-training toolbox and benchmark.
- [MMagic](https://github.com/open-mmlab/mmagic): Open**MM**Lab **A**dvanced, **G**enerative and **I**ntelligent **C**reation toolbox.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [Playground](https://github.com/open-mmlab/playground): A central hub for gathering and showcasing amazing projects built upon OpenMMLab.
