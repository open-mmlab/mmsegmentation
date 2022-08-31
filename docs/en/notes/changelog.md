# Changelog of v1.x

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
