# Tutorial 11: How to xxx

This tutorial collects answers to any `How to xxx with MMSegmentation`. Feel free to update this doc if you meet new questions about `How to` and find the answers!

## How to add a new metric

### Where to add a new metric

All metric function in MMSegMnetation is in [this file](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/core/evaluation/metrics.py). Therefore, if users would like to add a new metric, just add it in this file.

It all need to add a new if condition in [`eval_metrics`](https://github.com/open-mmlab/mmsegmentation/blob/1b24ad656f7c77bd79100bd7f35a00827043e53d/mmseg/core/evaluation/metrics.py#L256), which is for that the new metrics
can be called from `eval_metrics`.

### What is return from models when testing

The return from models is a dictionary with the `seg_pred` and `seg_logits` keys. The value of `seg_pred` is a list of length 1
which is the result for the input image, and the value of `seg_logits` is also a list of length 1 that is the class logits for the input. Therefore, the metric function input is this dictionary from models.

### What is return from metric function

The return from metric function is unlimited, and it will written in the log.
