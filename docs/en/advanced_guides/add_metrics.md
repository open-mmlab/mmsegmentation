# Add New Metrics

Here we show how to develop a new metric with an example of `CustomMetric` as the following.

1. Create a new file `mmseg/evaluation/metrics/custom_metric.py`.

   ```python
   import os.path as osp
   from collections import OrderedDict
   from typing import Dict, List, Optional, Sequence

   import numpy as np
   import torch
   from mmengine.dist import is_main_process
   from mmengine.evaluator import BaseMetric
   from mmengine.logging import MMLogger, print_log
   from mmengine.utils import mkdir_or_exist
   from PIL import Image
   from prettytable import PrettyTable

   from mmseg.registry import


   @METRICS.register_module()
   class CustomMetric(Metric):

       def __init__(self, arg1, arg2):
          pass

       def process(self, results, **kwargs) -> None:
           pass

       def compute_metrics(self, **kwargs) -> dict:
           pass

       def evaluate(self, **kwargs) -> dict:
           pass
   ```

   In the above example, `CustomMetric` is a subclass of `BaseMetric`. It has three methods: `process`, `compute_metrics` and `evaluate`.

   - `process` process one batch of data samples and predictions. The processed results are stored in `self.results` which will be used to compute the metrics after all the data samples are processed.

   - `compute_metrics` is used to compute the metrics from the processed results.

   - `evaluate` is an interface to compute the metrics and return the results. It will be called by the evaluator. In most cases, you don't need to override this method, but you can override it if you want to do some extra work.

2. Import the new metric in `mmseg/evaluation/metrics/__init__.py`.

   ```python
   from .custom_metric import CustomMetric
   __all__ = ['CustomMetric', ...]
   ```

3. Add the new metric to the config file.

   ```python
   metric = dict(type='CustomMetric', arg1=xxx, arg2=xxx)
   ```
