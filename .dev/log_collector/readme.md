# Log Collector

## Function

Automatically collect logs and write the result in a json file or markdown file.

If there are several `.log.json` files in one folder, Log Collector assumes that the `.log.json` files other than the first one are resume from the preceding `.log.json` file. Log Collector returns the result considering all `.log.json` files.

## Usage:

To use log collector, you need to write a config file to configure the log collector first.

For example:

example_config.py:

```python
# The work directory that contains folders that contains .log.json files.
work_dir = '../../work_dirs'
# The metric used to find the best evaluation.
metric = 'mIoU'

# **Don't specify the log_items and ignore_keywords at the same time.**
# Specify the log files we would like to collect in `log_items`.
# The folders specified should be the subdirectories of `work_dir`.
log_items = [
    'segformer_mit-b5_512x512_160k_ade20k_cnn_lr_with_warmup',
    'segformer_mit-b5_512x512_160k_ade20k_cnn_no_warmup_lr',
    'segformer_mit-b5_512x512_160k_ade20k_mit_trans_lr',
    'segformer_mit-b5_512x512_160k_ade20k_swin_trans_lr'
]
# Or specify `ignore_keywords`. The folders whose name contain one
# of the keywords in the `ignore_keywords` list(e.g., `'segformer'`)
# won't be collected.
# ignore_keywords = ['segformer']

# Other log items in .log.json that you want to collect.
# should not include metric.
other_info_keys = ["mAcc"]
# The output markdown file's name.
markdown_file ='markdowns/lr_in_trans.json.md'
# The output json file's name. (optional)
json_file = 'jsons/trans_in_cnn.json'
```

The structure of the work-dir directory should be like：

```text
├── work-dir
│   ├── folder1
│   │   ├── time1.log.json
│   │   ├── time2.log.json
│   │   ├── time3.log.json
│   │   ├── time4.log.json
│   ├── folder2
│   │   ├── time5.log.json
│   │   ├── time6.log.json
│   │   ├── time7.log.json
│   │   ├── time8.log.json
```

Then , cd to the log collector folder.

Now you can run log_collector.py by using command:

```bash
python log_collector.py ./example_config.py
```

The output markdown file is like:

| exp_num |                         method                          | mIoU best | best index | mIoU last | last index | last iter num |
| :-----: | :-----------------------------------------------------: | :-------: | :--------: | :-------: | :--------: | :-----------: |
|    1    | segformer_mit-b5_512x512_160k_ade20k_cnn_lr_with_warmup |  0.2776   |     10     |  0.2776   |     10     |    160000     |
|    2    |  segformer_mit-b5_512x512_160k_ade20k_cnn_no_warmup_lr  |  0.2802   |     10     |  0.2802   |     10     |    160000     |
|    3    |    segformer_mit-b5_512x512_160k_ade20k_mit_trans_lr    |  0.4943   |     11     |  0.4943   |     11     |    160000     |
|    4    |   segformer_mit-b5_512x512_160k_ade20k_swin_trans_lr    |  0.4883   |     11     |  0.4883   |     11     |    160000     |

The output json file is like:

```json
[
    {
        "method": "segformer_mit-b5_512x512_160k_ade20k_cnn_lr_with_warmup",
        "metric_used": "mIoU",
        "last_iter": 160000,
        "last eval": {
            "eval_index": 10,
            "mIoU": 0.2776,
            "mAcc": 0.3779
        },
        "best eval": {
            "eval_index": 10,
            "mIoU": 0.2776,
            "mAcc": 0.3779
        }
    },
    {
        "method": "segformer_mit-b5_512x512_160k_ade20k_cnn_no_warmup_lr",
        "metric_used": "mIoU",
        "last_iter": 160000,
        "last eval": {
            "eval_index": 10,
            "mIoU": 0.2802,
            "mAcc": 0.3764
        },
        "best eval": {
            "eval_index": 10,
            "mIoU": 0.2802,
            "mAcc": 0.3764
        }
    },
    {
        "method": "segformer_mit-b5_512x512_160k_ade20k_mit_trans_lr",
        "metric_used": "mIoU",
        "last_iter": 160000,
        "last eval": {
            "eval_index": 11,
            "mIoU": 0.4943,
            "mAcc": 0.6097
        },
        "best eval": {
            "eval_index": 11,
            "mIoU": 0.4943,
            "mAcc": 0.6097
        }
    },
    {
        "method": "segformer_mit-b5_512x512_160k_ade20k_swin_trans_lr",
        "metric_used": "mIoU",
        "last_iter": 160000,
        "last eval": {
            "eval_index": 11,
            "mIoU": 0.4883,
            "mAcc": 0.6061
        },
        "best eval": {
            "eval_index": 11,
            "mIoU": 0.4883,
            "mAcc": 0.6061
        }
    }
]
```
