# Add New Metrics

## Develop with the source code of MMSegmentation

Here we show how to develop a new metric with an example of `CustomMetric` as the following.

1. Create a new file `mmseg/evaluation/metrics/custom_metric.py`.

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

   In the above example, `CustomMetric` is a subclass of `BaseMetric`. It has three methods: `process`, `compute_metrics` and `evaluate`.

   - `process()` process one batch of data samples and predictions. The processed results are stored in `self.results` which will be used to compute the metrics after all the data samples are processed. Please refer to [MMEngine documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/en/design/evaluation.md) for more details.

   - `compute_metrics()` is used to compute the metrics from the processed results.

   - `evaluate()` is an interface to compute the metrics and return the results. It will be called by `ValLoop` or `TestLoop` in the `Runner`. In most cases, you don't need to override this method, but you can override it if you want to do some extra work.

   **Note:** You might find the details of calling `evaluate()` method in the `Runner` [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L366). The `Runner` is the executor of the training and testing process, you can find more details about it at the [engine document](./engine.md).

2. Import the new metric in `mmseg/evaluation/metrics/__init__.py`.

   ```python
   from .custom_metric import CustomMetric
   __all__ = ['CustomMetric', ...]
   ```

3. Add the new metric to the config file.

   ```python
   val_evaluator = dict(type='CustomMetric', arg1=xxx, arg2=xxx)
   test_evaluator = dict(type='CustomMetric', arg1=xxx, arg2=xxx)
   ```

## Develop with the released version of MMSegmentation

The above example shows how to develop a new metric with the source code of MMSegmentation. If you want to develop a new metric with the released version of MMSegmentation, you can follow the following steps.

1. Create a new file `/Path/to/metrics/custom_metric.py`, implement the `process`, `compute_metrics` and `evaluate` methods, `evaluate` method is optional.

2. Import the new metric in your code or config file.

   ```python
   from path.to.metrics import CustomMetric
   ```

   or

   ```python
   custom_imports = dict(imports=['/Path/to/metrics'], allow_failed_imports=False)

   val_evaluator = dict(type='CustomMetric', arg1=xxx, arg2=xxx)
   test_evaluator = dict(type='CustomMetric', arg1=xxx, arg2=xxx)
   ```
