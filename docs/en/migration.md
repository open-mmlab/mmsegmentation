# Migration from MMSegmentation 0.x

## Introduction

This guide describes the fundamental differences between MMSegmentation 0.x and MMSegmentation 1.x in terms of behaviors and the APIs, and how these all relate to your migration journey.

## New dependencies

MMSegmentation 1.x depends on some new packages, you can prepare a new clean environment and install again according to the [installation tutorial](get_started.md).
Or install the below packages manually.

1. [MMEngine](https://github.com/open-mmlab/mmengine): MMEngine is the core the OpenMMLab 2.0 architecture, and we splited many compentents unrelated to computer vision from MMCV to MMEngine.

2. [MMCV](https://github.com/open-mmlab/mmcv): The computer vision package of OpenMMLab. This is not a new dependency, but you need to upgrade it to above 2.0.0rc1 version.

3. [MMClassification](https://github.com/open-mmlab/mmclassification)(Optional): The  image classification toolbox and benchmark of OpenMMLab. This is not a new dependency, but you need to upgrade it to above 1.0.0rc0 version.

## Train launch

The main improvement of OpenMMLab 2.0 is releasing MMEngine which provides universal and powerful runner for unified interfaces to launch training jobs.

Compared with MMSeg0.x, MMSeg1.x provides fewer command line arguments in `tools/train.py`

<table class="docutils">
<tr>
<td>Function</td>
<td>Original</td>
<td>New</td>
</tr>
<tr>
<td>Train configuration file path</td>
<td>args.config</td>
<td>args.config</td>
</tr>
<tr>
<td>Directory to save logs and models</td>
<td>args.work_dir</td>
<td>args.work_dir</td>
</tr>
<tr>
<td>Loading pre-trained checkpoint</td>
<td>`args.load_from` or `cfg.load_from`</td>
<td>cfg.load_from</td>
</tr>
<tr>
<td>Resuming Train</td>
<td></td>
</tr>
<tr>
<td>Whether not to evaluate the checkpoint during training</td>
</tr>
<tr>
<td>Training device assignment</td>
</tr>

</table>

## Configuration file

### Model settings

No changes in `model.backbone`, `model.neck`, `model.decode_head` and `model.losses` fields.

Add `model.data_preprocessor` field to configure the `DataPreProcessor`, including:

- `mean`(Sequence, optional): The pixel mean of R, G, B channels. Defaults to None.

- `std`(Sequence, optional): The pixel standard deviation of R, G, B channels. Defaults to None.

- `size`(Sequence, optional): Fixed padding size.

- `size_divisor` (int, optional): The divisor of padded size.

- `seg_pad_val` (float, optional): Padding value of segmentation map. Default: 255.

- `padding_mode` (str): Type of padding. Default: constant.

  - constant: pads with a constant value, this value is specified with pad_val.

- `bgr_to_rgb` (bool): whether to convert image from BGR to RGB.Defaults to False.

- `rgb_to_bgr` (bool): whether to convert image from RGB to RGB. Defaults to False.

### Data settings

### Optimizer and Schedule settings

### Runtime settings
