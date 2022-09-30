# Package structures changes

This section is included if you are curious about what has changed between MMSeg 0.x and 1.x.

## `mmseg.apis`

OpenMMLab 2.0 tries to support unified interface for multitasking of Computer Vision,
and release more power and [`Runner`](https://github.com/open-mmlab/mmengine/blob/main/docs/en/design/runner.md),
so MMSeg 1.x removed modules in `train.py` and `test.py` renamed `init_segmentor` to `init_model` and `inference_segmentor` to `inference_model`
Here is the changes of `mmseg.apis`:

|       Function        | Changes                                         |
| :-------------------: | :---------------------------------------------- |
|   `init_segmentor`    | Renamed to `init_model`                         |
| `inference_segmentor` | Rename to `inference_segmentor`                 |
| `show_result_pyplot`  | Implemented based on `SegLocalVisualizer`       |
|     `train_model`     | Removed, use `runner.train` to train.           |
|   `multi_gpu_test`    | Removed, use `runner.test` to test.             |
|   `single_gpu_test`   | Removed, use `runner.test` to test.             |
|   `set_random_seed`   | Removed, use `mmengine.runner.set_random_seed`. |
|  `init_random_seed`   | Removed, use `mmengine.dist.sync_random_seed`.  |

## mmseg.datasets

OpenMMLab 2.0 defines the `BaseDataset` to function and interface of dataset, and MMSegmentation 1.x also follow this protocol and defines the `BaseSegDataset` inherted from `BaseDataset`. MMCV 2.x collects general data transforms for multiple tasks e.g. classification, detection, segmenation, so MMSegmentation 1.x uses these data transforms and removes them from mmseg.datasets

```
   Classes        | Changes                                         |
```

| :-------------------: | :---------------------------------------------- |
|   `CustomDataset`    | Renamed to `BaseDataset` and inherted from `BaseDataset` in MMEngine |
