# Log Collector

## Function

Automatically collect logs and write the result in a json file or markdown file.

If there are several .log.json files in one folder, Log Collector assumes that the .log.json files other than the first one is resume from the preceding .log.json file. Log Collector returns the result considering all .log.json files.

## Usage:

To use log collector, you need to write a config file to configure the log collector first.

For example:

example_config.py:

```python
# The work directory that contains folders that contains .log.json files.
work_dir = '../mmsegmentation-master/work_dirs'
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
other_info_keys = ['other_key']
# The output markdown file's name.
markdown_file ='markdowns/lr_in_trans.json.md'
# The output json file's name. (optional)
json_file = 'trans_in_cnn.json'
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

Then , you can run log_collector.py by using command:

```bash
python log_collector.py ./example_config.py
```

The output markdown file is like:

| exp_num |                            method                            | mIoU best | best index | mIoU last | last index |
| :-----: | :----------------------------------------------------------: | :-------: | :--------: | :-------: | :--------: |
|    1    |       deeplabv3plus_r101-d8_512x512_160k_ade20k_cnn_lr       |  0.4537   |     10     |  0.4537   |     10     |
|    2    | deeplabv3plus_r101-d8_512x512_160k_ade20k_cnn_with_warmup_lr |  0.4602   |     10     |  0.4602   |     10     |
|    3    |    deeplabv3plus_r101-d8_512x512_160k_ade20k_mit_trans_lr    |  0.4543   |     13     |  0.4543   |     13     |
|    4    |   deeplabv3plus_r101-d8_512x512_160k_ade20k_swin_trans_lr    |  0.4454   |     12     |  0.4454   |     12     |
