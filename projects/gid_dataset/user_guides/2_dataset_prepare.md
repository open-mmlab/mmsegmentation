## Gaofen Image Dataset (GID)

- GID 数据集可在[此处](https://x-ytong.github.io/project/GID.html)进行下载。
- GID 数据集包含 150 张 6800x7200 的大尺寸图像，标签为 RGB 标签。
- 根据[文献](https://ieeexplore.ieee.org/document/9343296/)，此处选择 15 张图像生成训练集和验证集，该 15 张图像包含了所有六类信息。所选的图像名称如下：

```None
  GF2_PMS1__L1A0000647767-MSS1
  GF2_PMS1__L1A0001064454-MSS1
  GF2_PMS1__L1A0001348919-MSS1
  GF2_PMS1__L1A0001680851-MSS1
  GF2_PMS1__L1A0001680853-MSS1
  GF2_PMS1__L1A0001680857-MSS1
  GF2_PMS1__L1A0001757429-MSS1
  GF2_PMS2__L1A0000607681-MSS2
  GF2_PMS2__L1A0000635115-MSS2
  GF2_PMS2__L1A0000658637-MSS2
  GF2_PMS2__L1A0001206072-MSS2
  GF2_PMS2__L1A0001471436-MSS2
  GF2_PMS2__L1A0001642620-MSS2
  GF2_PMS2__L1A0001787089-MSS2
  GF2_PMS2__L1A0001838560-MSS2
```

这里也提供了一个脚本来方便的筛选出15张图像，

```
python projects/gid_dataset/tools/dataset_converters/gid_select15imgFromAll.py {150 张图像的路径} {150 张标签的路径} {15 张图像的路径} {15 张标签的路径}
```

在选择出 15 张图像后，执行以下命令进行裁切及标签的转换，需要修改为您所存储 15 张图像及标签的路径。

```
python projects/gid_dataset/tools/dataset_converters/gid.py {15 张图像的路径} {15 张标签的路径}
```

完成裁切后的 GID 数据结构如下：

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── gid
│   │   ├── ann_dir
|   │   │   │   ├── train
|   │   │   │   ├── val
│   │   ├── img_dir
|   │   │   │   ├── train
|   │   │   │   ├── val

```
