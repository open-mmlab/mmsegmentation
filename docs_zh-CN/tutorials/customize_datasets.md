# 教程 2: 自定义数据集

## 通过重新组织数据来定制数据集

最简单的方法是将您的数据集进行转化，并组织成文件夹的形式。

如下的文件结构就是一个例子。

```none
├── data
│   ├── my_dataset
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   │   ├── zzz{img_suffix}
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   │   ├── zzz{seg_map_suffix}
│   │   │   ├── val

```

一个训练对将由 img_dir/ann_dir 里同样首缀的文件组成。

如果给定 `split` 参数，只有部分在 img_dir/ann_dir 里的文件会被加载。
我们可以对被包括在 split 文本里的文件指定前缀。

除此以外，一个 split 文本如下所示：

```none
xxx
zzz
```

只有

`data/my_dataset/img_dir/train/xxx{img_suffix}`,
`data/my_dataset/img_dir/train/zzz{img_suffix}`,
`data/my_dataset/ann_dir/train/xxx{seg_map_suffix}`,
`data/my_dataset/ann_dir/train/zzz{seg_map_suffix}` 将被加载。

注意：标注是跟图像同样的形状 (H, W)，其中的像素值的范围是 `[0, num_classes - 1]`。
您也可以使用 [pillow](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#palette) 的 `'P'` 模式去创建包含颜色的标注。

## 通过混合数据去定制数据集

MMSegmentation 同样支持混合数据集去训练。
当前它支持拼接 (concat) 和 重复 (repeat) 数据集。

### 重复数据集

我们使用 `RepeatDataset` 作为包装 (wrapper) 去重复数据集。
例如，假设原始数据集是 `Dataset_A`，为了重复它，配置文件如下：

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # 这是 Dataset_A 数据集的原始配置
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

### 拼接数据集

有2种方式去拼接数据集。

1. 如果您想拼接的数据集是同样的类型，但有不同的标注文件，
    您可以按如下操作去拼接数据集的配置文件：

    1. 您也许可以拼接两个标注文件夹 `ann_dir`

        ```python
        dataset_A_train = dict(
            type='Dataset_A',
            img_dir = 'img_dir',
            ann_dir = ['anno_dir_1', 'anno_dir_2'],
            pipeline=train_pipeline
        )
        ```

    2. 您也可以去拼接两个 `split` 文件列表

        ```python
        dataset_A_train = dict(
            type='Dataset_A',
            img_dir = 'img_dir',
            ann_dir = 'anno_dir',
            split = ['split_1.txt', 'split_2.txt'],
            pipeline=train_pipeline
        )
        ```

    3. 您也可以同时拼接 `ann_dir` 文件夹和 `split` 文件列表

        ```python
        dataset_A_train = dict(
            type='Dataset_A',
            img_dir = 'img_dir',
            ann_dir = ['anno_dir_1', 'anno_dir_2'],
            split = ['split_1.txt', 'split_2.txt'],
            pipeline=train_pipeline
        )
        ```

        在这样的情况下， `ann_dir_1` 和 `ann_dir_2` 分别对应于 `split_1.txt` 和 `split_2.txt`

2. 如果您想拼接不同的数据集，您可以如下去拼接数据集的配置文件：

    ```python
    dataset_A_train = dict()
    dataset_B_train = dict()

    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train = [
            dataset_A_train,
            dataset_B_train
        ],
        val = dataset_A_val,
        test = dataset_A_test
        )
    ```

一个更复杂的例子如下：分别重复 `Dataset_A` 和 `Dataset_B` N 次和 M 次，然后再去拼接重复后的数据集

```python
dataset_A_train = dict(
    type='RepeatDataset',
    times=N,
    dataset=dict(
        type='Dataset_A',
        ...
        pipeline=train_pipeline
    )
)
dataset_A_val = dict(
    ...
    pipeline=test_pipeline
)
dataset_A_test = dict(
    ...
    pipeline=test_pipeline
)
dataset_B_train = dict(
    type='RepeatDataset',
    times=M,
    dataset=dict(
        type='Dataset_B',
        ...
        pipeline=train_pipeline
    )
)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train = [
        dataset_A_train,
        dataset_B_train
    ],
    val = dataset_A_val,
    test = dataset_A_test
)

```
