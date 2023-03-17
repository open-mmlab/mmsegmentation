# Package structures changes

This section is included if you are curious about what has changed between MMSeg 0.x and 1.x.

<table>
<tr>
<td>MMSegmentation 0.x</td>
<td>MMSegmentation 1.x</td>
</tr>
<tr>
<td>mmseg.api</td>
<td>mmseg.api</td>
</tr>
<tr>
<td bgcolor=#fcf7f7>- mmseg.core</td>
<td bgcolor=#ecf4eb>+ mmseg.engine</td>
</tr>
<tr>
<td>mmseg.datasets</td>
<td>mmseg.datasets</td>
</tr>
<tr>
<td>mmseg.models</td>
<td>mmseg.models</td>
</tr>
<tr>
<td bgcolor=#fcf7f7>- mmseg.ops</td>
<td bgcolor=#ecf4eb>+ mmseg.structure</td>
</tr>
<tr>
<td>mmseg.utils</td>
<td>mmseg.utils</td>
</tr>
<tr>
<td></td>
<td bgcolor=#ecf4eb>+ mmseg.evaluation</td>
</tr>
<tr>
<td></td>
<td bgcolor=#ecf4eb>+ mmseg.registry</td>
<tr>
</table>

## Removed packages

### `mmseg.core`

In OpenMMLab 2.0, `core` package has been removed. `hooks` and `optimizers` of `core` are moved in `mmseg.engine`, and `evaluation` in `core` is mmseg.evaluation currently.

## `mmseg.ops`

`ops` package included `encoding` and `wrappers`, which are moved in `mmseg.models.utils`.

## Added packages

### `mmseg.engine`

OpenMMLab 2.0 adds a new foundational library for training deep learning, MMEngine. It servers as the training engine of all OpenMMLab codebases.
`engine` package of mmseg is some customized modules for semantic segmentation task, like `SegVisualizationHook` which works for visualizing segmentation mask.

### `mmseg.structure`

In OpenMMLab 2.0, we designed data structure for computer vision task, and in mmseg, we implements `SegDataSample` in `structure` package.

### `mmseg.evaluation`

We move all evaluation metric in `mmseg.evaluation`.

### `mmseg.registry`

We moved registry implementations for all kinds of modules in MMSegmentation in `mmseg.registry`.

## Modified packages

### `mmseg.apis`

OpenMMLab 2.0 tries to support unified interface for multitasking of Computer Vision, and releases much stronger [`Runner`](https://github.com/open-mmlab/mmengine/blob/main/docs/en/design/runner.md), so MMSeg 1.x removed modules in `train.py` and `test.py` renamed `init_segmentor` to `init_model` and `inference_segmentor` to `inference_model`.

Here is the changes of `mmseg.apis`:

|       Function        | Changes                                         |
| :-------------------: | :---------------------------------------------- |
|   `init_segmentor`    | Renamed to `init_model`                         |
| `inference_segmentor` | Rename to `inference_model`                     |
| `show_result_pyplot`  | Implemented based on `SegLocalVisualizer`       |
|     `train_model`     | Removed, use `runner.train` to train.           |
|   `multi_gpu_test`    | Removed, use `runner.test` to test.             |
|   `single_gpu_test`   | Removed, use `runner.test` to test.             |
|   `set_random_seed`   | Removed, use `mmengine.runner.set_random_seed`. |
|  `init_random_seed`   | Removed, use `mmengine.dist.sync_random_seed`.  |

### `mmseg.datasets`

OpenMMLab 2.0 defines the `BaseDataset` to function and interface of dataset, and MMSegmentation 1.x also follow this protocol and defines the `BaseSegDataset` inherited from `BaseDataset`. MMCV 2.x collects general data transforms for multiple tasks e.g. classification, detection, segmentation, so MMSegmentation 1.x uses these data transforms and removes them from mmseg.datasets.

|   Packages/Modules    | Changes                                                                                     |
| :-------------------: | :------------------------------------------------------------------------------------------ |
|   `mmseg.pipelines`   | Moved in `mmcv.transforms`                                                                  |
|    `mmseg.sampler`    | Moved in `mmengine.dataset.sampler`                                                         |
|    `CustomDataset`    | Renamed to `BaseSegDataset` and inherited from `BaseDataset` in MMEngine                    |
| `DefaultFormatBundle` | Replaced with `PackSegInputs`                                                               |
|  `LoadImageFromFile`  | Moved in `mmcv.transforms.LoadImageFromFile`                                                |
|   `LoadAnnotations`   | Moved in `mmcv.transforms.LoadAnnotations`                                                  |
|       `Resize`        | Moved in `mmcv.transforms` and split into `Resize`, `RandomResize` and `RandomChoiceResize` |
|     `RandomFlip`      | Moved in `mmcv.transforms.RandomFlip`                                                       |
|         `Pad`         | Moved in `mmcv.transforms.Pad`                                                              |
|      `Normalize`      | Moved in `mmcv.transforms.Normalize`                                                        |
|       `Compose`       | Moved in `mmcv.transforms.Compose`                                                          |
|    `ImageToTensor`    | Moved in `mmcv.transforms.ImageToTensor`                                                    |

### `mmseg.models`

`models` has not changed a lot, just added the `encoding` and `wrappers` from previous `mmseg.ops`
