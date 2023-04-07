# 包结构更改

本节包含您对 MMSeg 0.x 和 1.x 之间的变化可能感到好奇的内容。

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

## 已删除的包

### `mmseg.core`

在 OpenMMLab 2.0 中，`core` 包已被删除。`core` 的 `hooks` 和 `optimizers` 被移动到了 `mmseg.engine` 中，而 `core` 中的 `evaluation` 目前是 mmseg.evaluation。

## `mmseg.ops`

`ops` 包含 `encoding` 和 `wrappers`，它们被移到了 `mmseg.models.utils` 中。

## 增加的包

### `mmseg.engine`

OpenMMLab 2.0 增加了一个新的深度学习训练基础库 MMEngine。它是所有 OpenMMLab 代码库的训练引擎。
mmseg 的 `engine` 包是一些用于语义分割任务的定制模块，如 `SegVisualizationHook` 用于可视化分割掩膜。

### `mmseg.structure`

在 OpenMMLab 2.0 中，我们为计算机视觉任务设计了数据结构，在 mmseg 中，我们在 `structure` 包中实现了 `SegDataSample`。

### `mmseg.evaluation`

我们将所有评估指标都移动到了 `mmseg.evaluation` 中。

### `mmseg.registry`

我们将 MMSegmentation 中所有类型模块的注册实现移动到 `mmseg.registry` 中。

## 修改的包

### `mmseg.apis`

OpenMMLab 2.0 尝试支持计算机视觉的多任务统一接口，并发布了更强的 [`Runner`](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/design/runner.md)，因此 MMSeg 1.x 删除了 `train.py` 和 `test.py` 中的模块，并将 `init_segmentor` 重命名为 `init_model`，将 `inference_segmentor` 重命名为 `inference_model`。

以下是 `mmseg.apis` 的更改：

|         函数          | 变化                                           |
| :-------------------: | :--------------------------------------------- |
|   `init_segmentor`    | 重命名为 `init_model`                          |
| `inference_segmentor` | 重命名为 `inference_model`                     |
| `show_result_pyplot`  | 基于 `SegLocalVisualizer` 实现                 |
|     `train_model`     | 删除，使用 `runner.train` 训练。               |
|   `multi_gpu_test`    | 删除，使用 `runner.test` 测试。                |
|   `single_gpu_test`   | 删除，使用 `runner.test` 测试。                |
|   `set_random_seed`   | 删除，使用 `mmengine.runner.set_random_seed`。 |
|  `init_random_seed`   | 删除，使用 `mmengine.dist.sync_random_seed`。  |

### `mmseg.datasets`

OpenMMLab 2.0 将 `BaseDataset` 定义为数据集的函数和接口，MMSegmentation 1.x 也遵循此协议，并定义了从 `BaseDataset` 继承的 `BaseSegDataset`。MMCV 2.x 收集多种任务的通用数据转换，例如分类、检测、分割，因此 MMSegmentation 1.x 使用这些数据转换并将其从 mmseg.dataset 中删除。

|        包/模块        | 更改                                                                                 |
| :-------------------: | :----------------------------------------------------------------------------------- |
|   `mmseg.pipelines`   | 移动到 `mmcv.transforms` 中                                                          |
|    `mmseg.sampler`    | 移动到 `mmengine.dataset.sampler` 中                                                 |
|    `CustomDataset`    | 重命名为 `BaseSegDataset` 并从 MMEngine 中的 `BaseDataset` 继承                      |
| `DefaultFormatBundle` | 替换为 `PackSegInputs`                                                               |
|  `LoadImageFromFile`  | 移动到 `mmcv.transforms.LoadImageFromFile` 中                                        |
|   `LoadAnnotations`   | 移动到 `mmcv.transforms.LoadAnnotations` 中                                          |
|       `Resize`        | 移动到 `mmcv.transforms` 中并拆分为 `Resize`，`RandomResize` 和 `RandomChoiceResize` |
|     `RandomFlip`      | 移动到 `mmcv.transforms.RandomFlip` 中                                               |
|         `Pad`         | 移动到 `mmcv.transforms.Pad` 中                                                      |
|      `Normalize`      | 移动到 `mmcv.transforms.Normalize` 中                                                |
|       `Compose`       | 移动到 `mmcv.transforms.Compose` 中                                                  |
|    `ImageToTensor`    | 移动到 `mmcv.transforms.ImageToTensor` 中                                            |

### `mmseg.models`

`models` 没有太大变化，只是从以前的 `mmseg.ops` 添加了 `encoding` 和 `wrappers`
