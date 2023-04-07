# Changelog of v1.x

## v1.0.0(04/06/2023)

### Highlights

- Add Mapillary Vistas Datasets support to MMSegmentation Core Package ([#2576](https://github.com/open-mmlab/mmsegmentation/pull/2576))
- Support PIDNet ([#2609](https://github.com/open-mmlab/mmsegmentation/pull/2609))
- Support SegNeXt ([#2654](https://github.com/open-mmlab/mmsegmentation/pull/2654))

### Features

- Support calculating FLOPs of segmentors ([#2706](https://github.com/open-mmlab/mmsegmentation/pull/2706))
- Support multi-band image for Mosaic ([#2748](https://github.com/open-mmlab/mmsegmentation/pull/2748))
- Support dump segment prediction ([#2712](https://github.com/open-mmlab/mmsegmentation/pull/2712))

### Bug fix

- Fix format_result and fix prefix param in cityscape metric, and rename CitysMetric to CityscapesMetric ([#2660](https://github.com/open-mmlab/mmsegmentation/pull/2660))
- Support input gt seg map is not 2D ([#2739](https://github.com/open-mmlab/mmsegmentation/pull/2739))
- Fix accepting an unexpected argument `local-rank` in PyTorch 2.0 ([#2812](https://github.com/open-mmlab/mmsegmentation/pull/2812))

### Documentation

- Add Chinese version of various documentation ([#2673](https://github.com/open-mmlab/mmsegmentation/pull/2673), [#2702](https://github.com/open-mmlab/mmsegmentation/pull/2702), [#2703](https://github.com/open-mmlab/mmsegmentation/pull/2703), [#2701](https://github.com/open-mmlab/mmsegmentation/pull/2701), [#2722](https://github.com/open-mmlab/mmsegmentation/pull/2722), [#2733](https://github.com/open-mmlab/mmsegmentation/pull/2733), [#2769](https://github.com/open-mmlab/mmsegmentation/pull/2769), [#2790](https://github.com/open-mmlab/mmsegmentation/pull/2790), [#2798](https://github.com/open-mmlab/mmsegmentation/pull/2798))
- Update and refine various English documentation ([#2715](https://github.com/open-mmlab/mmsegmentation/pull/2715), [#2755](https://github.com/open-mmlab/mmsegmentation/pull/2755), [#2745](https://github.com/open-mmlab/mmsegmentation/pull/2745), [#2797](https://github.com/open-mmlab/mmsegmentation/pull/2797), [#2799](https://github.com/open-mmlab/mmsegmentation/pull/2799), [#2821](https://github.com/open-mmlab/mmsegmentation/pull/2821), [#2827](https://github.com/open-mmlab/mmsegmentation/pull/2827), [#2831](https://github.com/open-mmlab/mmsegmentation/pull/2831))
- Add deeplabv3 model structure documentation ([#2426](https://github.com/open-mmlab/mmsegmentation/pull/2426))
- Add custom metrics documentation ([#2799](https://github.com/open-mmlab/mmsegmentation/pull/2799))
- Add faq in dev-1.x branch ([#2765](https://github.com/open-mmlab/mmsegmentation/pull/2765))

### New Contributors

- @liuruiqiang made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2554
- @wangjiangben-hw made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2569
- @jinxianwei made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2557
- @KKIEEK made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2747
- @Renzhihan made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2765

## v1.0.0rc6(03/03/2023)

### Highlights

- Support MMSegInferencer ([#2413](https://github.com/open-mmlab/mmsegmentation/pull/2413), [#2658](https://github.com/open-mmlab/mmsegmentation/pull/2658))
- Support REFUGE dataset ([#2554](https://github.com/open-mmlab/mmsegmentation/pull/2554))

### Features

- Support auto import modules from registry ([#2481](https://github.com/open-mmlab/mmsegmentation/pull/2481))
- Replace numpy ascontiguousarray with torch contiguous to speed-up ([#2604](https://github.com/open-mmlab/mmsegmentation/pull/2604))
- Add browse_dataset.py tool ([#2649](https://github.com/open-mmlab/mmsegmentation/pull/2649))

### Bug fix

- Rename and Fix bug of projects HieraSeg ([#2565](https://github.com/open-mmlab/mmsegmentation/pull/2565))
- Add out_channels  in `CascadeEncoderDecoder` and update OCRNet and MobileNet v2 results ([#2656](https://github.com/open-mmlab/mmsegmentation/pull/2656))

### Documentation

- Add dataflow documentation of Chinese version ([#2652](https://github.com/open-mmlab/mmsegmentation/pull/2652))
- Add custmized runtime documentation of English version ([#2533](https://github.com/open-mmlab/mmsegmentation/pull/2533))
- Add documentation for visualizing feature map using wandb backend ([#2557](https://github.com/open-mmlab/mmsegmentation/pull/2557))
- Add documentation for benchmark results on NPU (HUAWEI Ascend) ([#2569](https://github.com/open-mmlab/mmsegmentation/pull/2569), [#2596](https://github.com/open-mmlab/mmsegmentation/pull/2596), [#2610](https://github.com/open-mmlab/mmsegmentation/pull/2610))
- Fix api name error in the migration doc ([#2601](https://github.com/open-mmlab/mmsegmentation/pull/2601))
- Refine projects documentation ([#2586](https://github.com/open-mmlab/mmsegmentation/pull/2586))
- Refine MMSegmentation documentation ([#2668](https://github.com/open-mmlab/mmsegmentation/pull/2668), [#2659](https://github.com/open-mmlab/mmsegmentation/pull/2659))

### New Contributors

- @zccjjj made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2548
- @liuruiqiang made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2554
- @wangjiangben-hw made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2569
- @jinxianwei made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2557

## v1.0.0rc5(02/01/2023)

### Bug fix

- Fix MaskFormer and Mask2Former when install mmdet from source ([#2532](https://github.com/open-mmlab/mmsegmentation/pull/2532))
- Support new fileio interface in `MMCV>=2.0.0rc4` ([#2543](https://github.com/open-mmlab/mmsegmentation/pull/2543))
- Fix ERFNet URL in dev-1.x branch ([#2537](https://github.com/open-mmlab/mmsegmentation/pull/2537))
- Fix misleading `List[Tensor]` types ([#2546](https://github.com/open-mmlab/mmsegmentation/pull/2546))
- Rename typing.py to typing_utils.py ([#2548](https://github.com/open-mmlab/mmsegmentation/pull/2548))

## v1.0.0rc4(01/30/2023)

### Highlights

- Support ISNet (ICCV'2021) in projects ([#2400](https://github.com/open-mmlab/mmsegmentation/pull/2400))
- Support HSSN (CVPR'2022) in projects ([#2444](https://github.com/open-mmlab/mmsegmentation/pull/2444))

### Features

- Add Gaussian Noise and Blur for biomedical data ([#2373](https://github.com/open-mmlab/mmsegmentation/pull/2373))
- Add BioMedicalRandomGamma ([#2406](https://github.com/open-mmlab/mmsegmentation/pull/2406))
- Add BioMedical3DPad ([#2383](https://github.com/open-mmlab/mmsegmentation/pull/2383))
- Add BioMedical3DRandomFlip ([#2404](https://github.com/open-mmlab/mmsegmentation/pull/2404))
- Add `gt_edge_map` field to SegDataSample ([#2466](https://github.com/open-mmlab/mmsegmentation/pull/2466))
- Support synapse dataset ([#2432](https://github.com/open-mmlab/mmsegmentation/pull/2432), [#2465](https://github.com/open-mmlab/mmsegmentation/pull/2465))
- Support Mapillary Vistas Dataset in projects ([#2484](https://github.com/open-mmlab/mmsegmentation/pull/2484))
- Switch order of `reduce_zero_label` and applying `label_map` ([#2517](https://github.com/open-mmlab/mmsegmentation/pull/2517))

### Documentation

- Add ZN Customized_runtime Doc ([#2502](https://github.com/open-mmlab/mmsegmentation/pull/2502))
- Add EN datasets.md ([#2464](https://github.com/open-mmlab/mmsegmentation/pull/2464))
- Fix minor typo in migration `package.md` ([#2518](https://github.com/open-mmlab/mmsegmentation/pull/2518))

### Bug fix

- Fix incorrect `img_shape` value assignment in RandomCrop ([#2469](https://github.com/open-mmlab/mmsegmentation/pull/2469))
- Fix inference api and support setting palette to SegLocalVisualizer ([#2475](https://github.com/open-mmlab/mmsegmentation/pull/2475))
- Unfinished label conversion from `-1` to `255` ([#2516](https://github.com/open-mmlab/mmsegmentation/pull/2516))

### New Contributors

- @blueyo0 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2373
- @Fivethousand5k made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2406
- @suyanzhou626 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2383
- @unrealMJ made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2400
- @Dominic23331 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2432
- @AI-Tianlong made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2444
- @morkovka1337 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2492
- @Leeinsn made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2404
- @siddancha made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2516

## v1.0.0rc3(31/12/2022)

### Highlights

- Support test time augmentation ([#2184](https://github.com/open-mmlab/mmsegmentation/pull/2184))
- Add 'Projects/' folder and the first example project ([#2412](https://github.com/open-mmlab/mmsegmentation/pull/2412))

### Features

- Add Biomedical 3D array random crop transform ([#2378](https://github.com/open-mmlab/mmsegmentation/pull/2378))

### Documentation

- Add Chinese version of config tutorial ([#2371](https://github.com/open-mmlab/mmsegmentation/pull/2371))
- Add Chinese version of train & test tutorial  ([#2355](https://github.com/open-mmlab/mmsegmentation/pull/2355))
- Add Chinese version of overview ([(#2397)](https://github.com/open-mmlab/mmsegmentation/pull/2397)))
- Add Chinese version of get_started ([#2417](https://github.com/open-mmlab/mmsegmentation/pull/2417))
- Add datasets in Chinese ([#2387](https://github.com/open-mmlab/mmsegmentation/pull/2387))
- Add dataflow document ([#2403](https://github.com/open-mmlab/mmsegmentation/pull/2403))
- Add pspnet model structure graph ([#2437](https://github.com/open-mmlab/mmsegmentation/pull/2437))
- Update some content of engine Chinese documentation ([#2341](https://github.com/open-mmlab/mmsegmentation/pull/2341))
- Update TTA to migration documentation ([#2335](https://github.com/open-mmlab/mmsegmentation/pull/2335))

### Bug fix

- Remove dependency mmdet when do not use MaskFormerHead and MMDET_Mask2FormerHead ([#2448](https://github.com/open-mmlab/mmsegmentation/pull/2448))

### Enhancement

- Add torch1.13 checking in CI ([#2402](https://github.com/open-mmlab/mmsegmentation/pull/2402))
- Fix pytorch version for merge stage test  ([#2449](https://github.com/open-mmlab/mmsegmentation/pull/2449))

## v1.0.0rc2(6/12/2022)

### Highlights

- Support MaskFormer ([#2215](https://github.com/open-mmlab/mmsegmentation/pull/2215))
- Support Mask2Former ([#2255](https://github.com/open-mmlab/mmsegmentation/pull/2255))

### Features

- Add ResizeShortestEdge transform ([#2339](https://github.com/open-mmlab/mmsegmentation/pull/2339))
- Support padding in data pre-processor for model testing([#2290](https://github.com/open-mmlab/mmsegmentation/pull/2290))
- Fix the problem of post-processing not removing padding ([#2367](https://github.com/open-mmlab/mmsegmentation/pull/2367))

### Bug fix

- Fix links in README ([#2024](https://github.com/open-mmlab/mmsegmentation/pull/2024))
- Fix swin load state_dict ([#2304](https://github.com/open-mmlab/mmsegmentation/pull/2304))
- Fix typo of BaseSegDataset docstring ([#2322](https://github.com/open-mmlab/mmsegmentation/pull/2322))
- Fix the bug in the visualization step ([#2326](https://github.com/open-mmlab/mmsegmentation/pull/2326))
- Fix ignore class id from -1 to 255 in BaseSegDataset ([#2332](https://github.com/open-mmlab/mmsegmentation/pull/2332))
- Fix KNet IterativeDecodeHead bug ([#2334](https://github.com/open-mmlab/mmsegmentation/pull/2334))
- Add input argument for datasets ([#2379](https://github.com/open-mmlab/mmsegmentation/pull/2379))
- Fix typo in warning on binary classification ([#2382](https://github.com/open-mmlab/mmsegmentation/pull/2382))

### Enhancement

- Fix ci for 1.x ([#2011](https://github.com/open-mmlab/mmsegmentation/pull/2011), [#2019](https://github.com/open-mmlab/mmsegmentation/pull/2019))
- Fix lint and pre-commit hook ([#2308](https://github.com/open-mmlab/mmsegmentation/pull/2308))
- Add `data` string in .gitignore file in dev-1.x branch ([#2336](https://github.com/open-mmlab/mmsegmentation/pull/2336))
- Make scipy as a default dependency in runtime ([#2362](https://github.com/open-mmlab/mmsegmentation/pull/2362))
- Delete mmcls in runtime.txt ([#2368](https://github.com/open-mmlab/mmsegmentation/pull/2368))

### Documentation

- Update configuration documentation ([#2048](https://github.com/open-mmlab/mmsegmentation/pull/2048))
- Update inference documentation ([#2052](https://github.com/open-mmlab/mmsegmentation/pull/2052))
- Update train test documentation ([#2061](https://github.com/open-mmlab/mmsegmentation/pull/2061))
- Update get started documentatin ([#2148](https://github.com/open-mmlab/mmsegmentation/pull/2148))
- Update transforms documentation ([#2088](https://github.com/open-mmlab/mmsegmentation/pull/2088))
- Add MMEval projects like in README ([#2259](https://github.com/open-mmlab/mmsegmentation/pull/2259))
- Translate the visualization.md ([#2298](https://github.com/open-mmlab/mmsegmentation/pull/2298))

## v1.0.0rc1 (2/11/2022)

### Highlights

- Support PoolFormer ([#2191](https://github.com/open-mmlab/mmsegmentation/pull/2191))
- Add Decathlon dataset ([#2227](https://github.com/open-mmlab/mmsegmentation/pull/2227))

### Features

- Add BioMedical data loading ([#2176](https://github.com/open-mmlab/mmsegmentation/pull/2176))
- Add LIP dataset ([#2251](https://github.com/open-mmlab/mmsegmentation/pull/2251))
- Add `GenerateEdge` data transform ([#2210](https://github.com/open-mmlab/mmsegmentation/pull/2210))

### Bug fix

- Fix segmenter-vit-s_fcn config ([#2037](https://github.com/open-mmlab/mmsegmentation/pull/2037))
- Fix binary segmentation ([#2101](https://github.com/open-mmlab/mmsegmentation/pull/2101))
- Fix MMSegmentation colab demo ([#2089](https://github.com/open-mmlab/mmsegmentation/pull/2089))
- Fix ResizeToMultiple transform ([#2185](https://github.com/open-mmlab/mmsegmentation/pull/2185))
- Use SyncBN in mobilenet_v2 ([#2198](https://github.com/open-mmlab/mmsegmentation/pull/2198))
- Fix typo in installation ([#2175](https://github.com/open-mmlab/mmsegmentation/pull/2175))
- Fix typo in visualization.md ([#2116](https://github.com/open-mmlab/mmsegmentation/pull/2116))

### Enhancement

- Add mim extras_requires in setup.py ([#2012](https://github.com/open-mmlab/mmsegmentation/pull/2012))
- Fix CI ([#2029](https://github.com/open-mmlab/mmsegmentation/pull/2029))
- Remove ops module ([#2063](https://github.com/open-mmlab/mmsegmentation/pull/2063))
- Add pyupgrade pre-commit hook ([#2078](https://github.com/open-mmlab/mmsegmentation/pull/2078))
- Add `out_file` in `add_datasample` of `SegLocalVisualizer` to directly save image ([#2090](https://github.com/open-mmlab/mmsegmentation/pull/2090))
- Upgrade pre commit hooks ([#2154](https://github.com/open-mmlab/mmsegmentation/pull/2154))
- Ignore test timm in CI when torch\<1.7 ([#2158](https://github.com/open-mmlab/mmsegmentation/pull/2158))
- Update requirements ([#2186](https://github.com/open-mmlab/mmsegmentation/pull/2186))
- Fix Windows platform CI ([#2202](https://github.com/open-mmlab/mmsegmentation/pull/2202))

### Documentation

- Add `Overview` documentation ([#2042](https://github.com/open-mmlab/mmsegmentation/pull/2042))
- Add `Evaluation` documentation ([#2077](https://github.com/open-mmlab/mmsegmentation/pull/2077))
- Add `Migration` documentation ([#2066](https://github.com/open-mmlab/mmsegmentation/pull/2066))
- Add `Structures` documentation ([#2070](https://github.com/open-mmlab/mmsegmentation/pull/2070))
- Add `Structures` ZN documentation ([#2129](https://github.com/open-mmlab/mmsegmentation/pull/2129))
- Add `Engine` ZN documentation ([#2157](https://github.com/open-mmlab/mmsegmentation/pull/2157))
- Update `Prepare datasets` and `Visualization` doc ([#2054](https://github.com/open-mmlab/mmsegmentation/pull/2054))
- Update `Models` documentation ([#2160](https://github.com/open-mmlab/mmsegmentation/pull/2160))
- Update `Add New Modules` documentation ([#2067](https://github.com/open-mmlab/mmsegmentation/pull/2067))
- Fix the installation commands in get_started.md ([#2174](https://github.com/open-mmlab/mmsegmentation/pull/2174))
- Add MMYOLO to README.md ([#2220](https://github.com/open-mmlab/mmsegmentation/pull/2220))

## v1.0.0rc0 (31/8/2022)

We are excited to announce the release of MMSegmentation 1.0.0rc0.
MMSeg 1.0.0rc0 is the first version of MMSegmentation 1.x, a part of the OpenMMLab 2.0 projects.
Built upon the new [training engine](https://github.com/open-mmlab/mmengine),
MMSeg 1.x unifies the interfaces of dataset, models, evaluation, and visualization with faster training and testing speed.

### Highlights

1. **New engines** MMSeg 1.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a general and powerful runner that allows more flexible customizations and significantly simplifies the entrypoints of high-level interfaces.

2. **Unified interfaces** As a part of the OpenMMLab 2.0 projects, MMSeg 1.x unifies and refactors the interfaces and internal logics of train, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.0 projects share the same design in those interfaces and logics to allow the emergence of multi-task/modality algorithms.

3. **Faster speed** We optimize the training and inference speed for common models.

4. **New features**:

   - Support TverskyLoss function

5. **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmsegmentation.readthedocs.io/en/1.x/).

### Breaking Changes

We briefly list the major breaking changes here.
We will update the [migration guide](../migration.md) to provide complete details and migration instructions.

#### Training and testing

- MMSeg 1.x runs on PyTorch>=1.6. We have deprecated the support of PyTorch 1.5 to embrace the mixed precision training and other new features since PyTorch 1.6. Some models can still run on PyTorch 1.5, but the full functionality of MMSeg 1.x is not guaranteed.

- MMSeg 1.x uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of dataset, model, evaluation, and visualizer. Therefore, MMSeg 1.x no longer maintains the building logics of those modules in `mmseg.train.apis` and `tools/train.py`. Those code have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py). Please refer to the [migration guide of Runner in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for more details.

- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic as that in training scripts to build the runner.

- The execution points of hooks in the new Runner have been enriched to allow more flexible customization. Please refer to the [migration guide of Hook in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/hook.html) for more details.

- Learning rate and momentum scheduling has been migrated from `Hook` to `Parameter Scheduler` in MMEngine. Please refer to the [migration guide of Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html) for more details.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structures to ease the understanding of the components in runner. Users can read the [config example of mmseg](../user_guides/config.md) or refer to the [migration guide in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for migration details.
- The file names of configs and models are also refactored to follow the new rules unified across OpenMMLab 2.0 projects. Please refer to the [user guides of config](../user_guides/1_config.md) for more details.

#### Components

- Dataset
- Data Transforms
- Model
- Evaluation
- Visualization

### Improvements

- Support mixed precision training of all the models. However, some models may got Nan results due to some numerical issues. We will update the documentation and list their results (accuracy of failure) of mixed precision training.

### Bug Fixes

- Fix several config file errors [#1994](https://github.com/open-mmlab/mmsegmentation/pull/1994)

### New Features

1. Support data structures and encapsulating `seg_logits` in data samples, which can be return from models to support more common evaluation metrics.

### Ongoing changes

1. Test-time augmentation: which is supported in MMSeg 0.x is not implemented in this version due to limited time slot. We will support it in the following releases with a new and simplified design.

2. Inference interfaces: a unified inference interfaces will be supported in the future to ease the use of released models.

3. Interfaces of useful tools that can be used in notebook: more useful tools that implemented in the `tools` directory will have their python interfaces so that they can be used through notebook and in downstream libraries.

4. Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMSeg 1.x.
