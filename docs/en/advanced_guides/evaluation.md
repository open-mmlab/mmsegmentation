# Evaluation

MMSegmentation implements [IoUMetric](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/evaluation/metrics/iou_metric.py) and [CitysMetric](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/evaluation/metrics/citys_metric.py) for evaluating the performance of models, based on the [BaseMetric](https://github.com/open-mmlab/mmengine/blob/main/mmengine/evaluator/metric.py) provided by [MMEngine](https://github.com/open-mmlab/mmengine).

## IOUMetric

`IoUMetric` implements the IoU metric calculation, the core two methods of `IoUMetric` are `process` and `compute_metrics`.

- `process` method processes one batch of data and data_samples.
- `compute_metrics` method computes the metrics from processed results.

### process

pass

### compute_metrics

pass
