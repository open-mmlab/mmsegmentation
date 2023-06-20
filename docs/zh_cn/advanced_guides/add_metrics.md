# 新增评测指标

## 使用 MMSegmentation 的源代码进行开发

在这里，我们用 `CustomMetric` 作为例子来展示如何开发一个新的评测指标。

1. 创建一个新文件 `mmseg/evaluation/metrics/custom_metric.py`。

   ```python
   from typing import List, Sequence

   from mmengine.evaluator import BaseMetric

   from mmseg.registry import METRICS


   @METRICS.register_module()
   class CustomMetric(BaseMetric):

       def __init__(self, arg1, arg2):
           """
           The metric first processes each batch of data_samples and predictions,
           and appends the processed results to the results list. Then it
           collects all results together from all ranks if distributed training
           is used. Finally, it computes the metrics of the entire dataset.
           """

       def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
           pass

       def compute_metrics(self, results: list) -> dict:
           pass

       def evaluate(self, size: int) -> dict:
           pass
   ```

   在上面的示例中，`CustomMetric` 是 `BaseMetric` 的子类。它有三个方法：`process`，`compute_metrics` 和 `evaluate`。

   - `process()` 处理一批数据样本和预测。处理后的结果需要显示地传给 `self.results` ，将在处理所有数据样本后用于计算指标。更多细节请参考 [MMEngine 文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/design/evaluation.md)

   - `compute_metrics()` 用于从处理后的结果中计算指标。

   - `evaluate()` 是一个接口，用于计算指标并返回结果。它将由 `ValLoop` 或 `TestLoop` 在 `Runner` 中调用。在大多数情况下，您不需要重写此方法，但如果您想做一些额外的工作，可以重写它。

   **注意：** 您可以在[这里](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L366) 找到 `Runner` 调用 `evaluate()` 方法的过程。`Runner` 是训练和测试过程的执行器，您可以在[训练引擎文档](./engine.md)中找到有关它的详细信息。

2. 在 `mmseg/evaluation/metrics/__init__.py` 中导入新的指标。

   ```python
   from .custom_metric import CustomMetric
   __all__ = ['CustomMetric', ...]
   ```

3. 在配置文件中设置新的评测指标

   ```python
   val_evaluator = dict(type='CustomMetric', arg1=xxx, arg2=xxx)
   test_evaluator = dict(type='CustomMetric', arg1=xxx, arg2=xxx)
   ```

## 使用发布版本的 MMSegmentation 进行开发

上面的示例展示了如何使用 MMSegmentation 的源代码开发新指标。如果您想使用 MMSegmentation 的发布版本开发新指标，可以按照以下步骤操作。

1. 创建一个新文件 `/Path/to/metrics/custom_metric.py`，实现 `process`，`compute_metrics` 和 `evaluate` 方法，`evaluate` 方法是可选的。

2. 在代码或配置文件中导入新的指标。

   ```python
   from path.to.metrics import CustomMetric
   ```

   或者

   ```python
   custom_imports = dict(imports=['/Path/to/metrics'], allow_failed_imports=False)

   val_evaluator = dict(type='CustomMetric', arg1=xxx, arg2=xxx)
   test_evaluator = dict(type='CustomMetric', arg1=xxx, arg2=xxx)
   ```
