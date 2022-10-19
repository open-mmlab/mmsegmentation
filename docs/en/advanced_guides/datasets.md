# Dataset

In this document, we will introduce the design of dataset in MMSegmentation and how users can design their own dataset.

In the 1.x version of MMSegmentation, all datasets are inherited from `BaseSegDataset`.
The function of dataset is loading the `data_list` (please refer [basesegdataset.py](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/datasets/basesegdataset.py) for more information about `data_list`).
In `__getitem__`, `prepare_data` is called to get the processed data.
In `prepare_data`, data loading pipeline consists of the following steps:

1. fetch the data info by index, implemented by `get_data_info`
2. apply data transforms to data, implemented by `pipeline`

If you want to implement a new dataset class, you only need to implement `load_data_list` function. We also encourage users to use the original data loading logic provided by `BaseDataset`.
If the default loading logic is difficult to meet your needs, you can overwrite the `__getitem__` interface to implement your data loading logic.

The structure of this guide is as follows:

- [Dataset](#dataset)
  - [BaseSegDataset](#BaseSegDataset)
  - [Differences between CustomDataset(v0.x)](<#Differences-between-CustomDataset(v0.x)>)
  - [Design your own dataset](#design-your-own-dataset)

## BaseSegDataset

`BaseSegDataset` is designed for supervised learning semantic segmentation models that both need images and annotations. The directory structure is shown below.

```none
├── data
│   ├── my_dataset
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   ├── val
│   │   │   │   ├── zzz{img_suffix}
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   ├── val
│   │   │   │   ├── zzz{seg_map_suffix}

```

The image and ground truth annotation pair of BaseSegDataset should be of the same except suffix.
A valid image and ground truth annotation filename pair should be like `xxx{img_suffix}` and `xxx{seg_map_suffix}` (extension is also included
in the suffix). If split is given, then `xxx` is specified in txt file.
Otherwise, all files in `img_dir/`and `ann_dir` will be loaded.

In this dataset, we overwrite `load_data_list` function to add certain keys in `data_list` such as `seg_map_path` and `reduce_zero_label` for train/test process.

```python
def load_data_list(self) -> List[dict]:
    """Load annotation from directory or annotation file.

    Returns:
        list[dict]: All data info of dataset.
    """
    data_list = []
    img_dir = self.data_prefix.get('img_path', None)
    ann_dir = self.data_prefix.get('seg_map_path', None)
    if osp.isfile(self.ann_file):
        lines = mmengine.list_from_file(
            self.ann_file, file_client_args=self.file_client_args)
        for line in lines:
            img_name = line.strip()
            data_info = dict(
                img_path=osp.join(img_dir, img_name + self.img_suffix))
            if ann_dir is not None:
                seg_map = img_name + self.seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
    else:
        for img in self.file_client.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True):
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
        data_list = sorted(data_list, key=lambda x: x['img_path'])
    return data_list
```

## Differences between CustomDataset(v0.x)

In MMSegmentation \< 1.x, it uses [`CustomDataset`](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/custom.py#L19) as basic dataset class in MMSegmentation.

`BaseSegDataset` is inherited from `BaseDataset` in MMEngine while `CustomDataset` is inherited from `Dataset` in official PyTorch. The differences between `CustomDataset` and `BaseSegDataset` are lied below:

- `BaseSegDataset` removes all methods about evaluations in `CustomDataset`, such as `format_results`, `pre_eval`, `get_gt_seg_map_by_idx`, `get_gt_seg_maps` and `evaluate`.
- `BaseSegDataset` replaces method `load_annotations` to `load_data_list`.
- `BaseSegDataset` replaces member variable `img_infos` and `split` to `data_list` and `ann_file`, respectively.
- `BaseSegDataset` integrates `CLASSES` and `PALETTE` into two fields of `METAINFO` dict.

## Design your own dataset

You may refer to [tutorial in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/basedataset.md).
