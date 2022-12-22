# RF-Next: Efficient Receptive Field Search for CNN

> [RF-Next: Efficient Receptive Field Search for Convolutional Neural Networks](http://mftp.mmcheng.net/Papers/22TPAMI-ActionSeg.pdf)

<!-- [ALGORITHM] -->

## Abstract

Temporal/spatial receptive fields of models play an important role in sequential/spatial tasks. Large receptive fields facilitate long-term relations, while small receptive fields help to capture the local details. Existing methods construct models with hand-designed receptive fields in layers. Can we effectively search for receptive field combinations to replace hand-designed patterns? To answer this question, we propose to find better receptive field combinations through a global-to-local search scheme. Our search scheme exploits both global search to find the coarse combinations and local search to get the refined receptive field combinations further. The global search finds possible coarse combinations other than human-designed patterns. On top of the global search, we propose an expectation-guided iterative local search scheme to refine combinations effectively. Our RF-Next models, plugging receptive field search to various models, boost the performance on many tasks, e.g., temporal action segmentation, object detection, instance segmentation, and speech synthesis.
The source code is publicly available on [http://mmcheng.net/rfnext](http://mmcheng.net/rfnext).

## Results and Models

### ConvNext on ADE20K

|   Backbone    | Method  |     RFNext      | Crop Size | Lr Schd | mIoU  | mIoU (ms+flip) |                                                                config                                                                 |                                                                                                                                                                                           download                                                                                                                                                                                           |
| :-----------: | :-----: | :-------------: | :-------: | :-----: | :---: | :------------: | :-----------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  ConvNeXt-B   | UPerNet |       NO        |  640x640  | 160000  | 52.13 |     52.66      | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k/upernet_convnext_tiny_fp16_512x512_160k_ade20k_20220227_124553-cad485de.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k/upernet_convnext_tiny_fp16_512x512_160k_ade20k_20220227_124553.log.json) |
| RF-ConvNext-B | UperNet |  Single-Branch  |  640x640  | 160000  | 52.51 |     53.12      |                                                     [search](<>) \| [retrain](<>)                                                     |                                                                                                                                                                                   [model](<>) \| [log](<>)                                                                                                                                                                                   |
| RF-ConvNext-B | UperNet | Multiple-Branch |  640x640  | 160000  | 52.49 |     53.00      |                                                     [search](<>) \| [retrain](<>)                                                     |                                                                                                                                                                                   [model](<>) \| [log](<>)                                                                                                                                                                                   |

### DeepLabV3 on ADE20K

|  Backbone   |  Method   |     RFNext      | Crop Size | Lr Schd | mIoU  | mIoU (ms+flip) |                                                            config                                                            |                                                                                                                                                                        download                                                                                                                                                                        |
| :---------: | :-------: | :-------------: | :-------: | :-----: | :---: | :------------: | :--------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  R-101-D8   | DeepLabV3 |       NO        |  512x512  |  80000  | 44.08 |     45.19      | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k/deeplabv3_r101-d8_512x512_80k_ade20k_20200615_021256-d89c7fa4.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k/deeplabv3_r101-d8_512x512_80k_ade20k_20200615_021256.log.json) |
| RF-R-101-D8 | DeepLabV3 |  Single-Branch  |  512x512  |  80000  | 44.65 |     45.65      |                                                [search](<>) \| [retrain](<>)                                                 |                                                                                                                                                                [model](<>) \| [log](<>)                                                                                                                                                                |
| RF-R-101-D8 | DeepLabV3 | Multiple-Branch |  512x512  |  80000  | 44.77 |     45.86      |                                                [search](<>) \| [retrain](<>)                                                 |                                                                                                                                                                [model](<>) \| [log](<>)                                                                                                                                                                |

### DeepLabV3 on Pascal VOC 2012 + Aug

|  Backbone   |  Method   |     RFNext      | Crop Size | Lr Schd | mIoU  | mIoU (ms+flip) |                                                            config                                                             |                                                                                                                                                                          download                                                                                                                                                                          |
| :---------: | :-------: | :-------------: | :-------: | :-----: | :---: | :------------: | :---------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50-D8   | DeepLabV3 |       NO        |  512x512  |  20000  | 76.17 |     77.42      | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug/deeplabv3_r50-d8_512x512_20k_voc12aug_20200617_010906-596905ef.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug/deeplabv3_r50-d8_512x512_20k_voc12aug_20200617_010906.log.json) |
| RF-R-50-D8  | DeepLabV3 |  Single-Branch  |  512x512  |  20000  | 77.48 |     78.83      |                                                 [search](<>) \| [retrain](<>)                                                 |                                                                                                                                                                  [model](<>) \| [log](<>)                                                                                                                                                                  |
| RF-R-50-D8  | DeepLabV3 | Multiple-Branch |  512x512  |  20000  | 77.60 |     78.67      |                                                 [search](<>) \| [retrain](<>)                                                 |                                                                                                                                                                  [model](<>) \| [log](<>)                                                                                                                                                                  |
|  R-101-D8   | DeepLabV3 |       NO        |  512x512  |  20000  | 78.70 |     79.95      | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k.py)  |   [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k/deeplabv3_r101-d8_512x512_80k_ade20k_20200615_021256-d89c7fa4.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k/deeplabv3_r101-d8_512x512_80k_ade20k_20200615_021256.log.json)   |
| RF-R-101-D8 | DeepLabV3 |  Single-Branch  |  512x512  |  20000  | 79.51 |     80.82      |                                                 [search](<>) \| [retrain](<>)                                                 |                                                                                                                                                                  [model](<>) \| [log](<>)                                                                                                                                                                  |
| RF-R-101-D8 | DeepLabV3 | Multiple-Branch |  512x512  |  20000  | 79.55 |     80.68      |                                                 [search](<>) \| [retrain](<>)                                                 |                                                                                                                                                                  [model](<>) \| [log](<>)                                                                                                                                                                  |

**Note:** The performance of multi-branch models listed above are evaluated during searching to save computional cost, retraining would achieve similar or better performance.

## Configs

If you want to search receptive fields on an existing model, you need to define a `RFSearchHook` in the `custom_hooks` of config file.

```python
custom_hooks = [
    dict(
        type='RFSearchHook',
        mode='search',
        rfstructure_file=None,
        verbose=True,
        by_epoch=False,
        config=dict(
            search=dict(
                step=0,
                max_step=64001,
                search_interval=8000,
                exp_rate=0.5,
                init_alphas=0.01,
                mmin=1,
                mmax=64,
                num_branches=3,
                skip_layer=[]))
                # For the models with auxiliary heads,
                # we recommend skipping the layers used by auxiliary heads.
                # You can add these layers to `skip_layer`.
        ),
]
```

**Note:** The `by_epoch` should be set as `False` in mmsegmentation.

Arguments:

- `max_step`: The maximum number of steps to update the structures.
- `search_interval`: The interval (epoch) between two updates.
- `exp_rate`:  The controller of the sparsity of search space. For a conv with an initial dilation rate of `D`, dilation rates will be sampled with an interval of `exp_rate * D`.
- `num_branches`: The controller of the size of search space (the number of branches). If you set `S=3`, the dilations are `[D - exp_rate * D, D, D + exp_rate * D]` for three branches. If you set `num_branches=2`, the dilations are `[D - exp_rate * D, D + exp_rate * D]`. With `num_branches=2`, you can achieve similar performance with less MEMORY and FLOPS.
- `skip_layer`: The modules in skip_layer will be ignored during the receptive field search.

## Training

### 1. Searching Jobs

You can launch searching jobs by using config files with prefix `rfnext_search`. The json files of searched structures will be saved to `work_dir`.

If you want to further search receptive fields upon a searched structure, please set `rfsearch_cfg.rfstructure_file` in config file to the corresponding json file.

### 2. Training Jobs

Setting `rfsearch_cfg.rfstructure_file` to the searched structure file (.json) and setting `rfsearch_cfg.mode` to `fixed_single_branch` or `fixed_multi_branch`, you can retrain a model with the searched structure.
You can launch fixed_single_branch/fixed_multi_branch training jobs by using config files with prefix `rfnext_fixed_single_branch` or `rfnext_fixed_multi_branch`.

Note that the models after the searching stage is ready a `fixed_multi_branch` version, which achieves better performance than `fixed_single_branch`, without any retraining.

## Inference

`rfsearch_cfg.rfstructure_file` and `rfsearch_cfg.mode` should be set for inferencing stage.

**Note:For the models trained with modes of `fixed_single_branch` or `fixed_multi_branch`, you can just use the training config for inferencing.**
**But If you want to inference the models trained with the mode of `search`, please use the config with prefix of `rfnext_fixed_multi_branch` to inference the models. (Otherwise, you should set `rfsearch_cfg.mode` to `fixed_multi_branch` and set the searched rfstructure_file.)**

## Citation

```
@article{gao2022rfnext,
title={RF-Next: Efficient Receptive Field Search for Convolutional Neural Networks},
author={Gao, Shanghua and Li, Zhong-Yu and Han, Qi and Cheng, Ming-Ming and Wang, Liang},
journal=TPAMI,
year={2022}
}

@inproceedings{gao2021global2local,
  title     = {Global2Local: Efficient Structure Search for Video Action Segmentation},
  author    = {Gao, Shanghua and Han, Qi and Li, Zhong-Yu and Peng, Pai and Wang, Liang and Cheng, Ming-Ming},
  booktitle = CVPR,
  year      = {2021}
}
```
