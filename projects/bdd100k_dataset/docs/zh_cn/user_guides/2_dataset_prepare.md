## BDD100K

- 可以从[官方网站](https://bdd-data.berkeley.edu/) 下载 BDD100K数据集（语义分割任务主要是10K数据集），按照官网要求注册并登陆后，数据可以在[这里](https://bdd-data.berkeley.edu/portal.html#download)找到。

- 图像数据对应的名称是是`10K Images`, 语义分割标注对应的名称是`Segmentation`

- 下载后，可以使用以下代码进行解压

  ```bash
  unzip ~/bdd100k_images_10k.zip -d ~/mmsegmentation/data/
  unzip ~/bdd100k_sem_seg_labels_trainval.zip -d ~/mmsegmentation/data/
  ```

就可以得到以下文件结构了：

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── bdd100k
│   │   ├── images
│   │   │   └── 10k
|   │   │   │   ├── test
|   │   │   │   ├── train
|   │   │   │   └── val
│   │   └── labels
│   │   │   └── sem_seg
|   │   │   │   ├── colormaps
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── masks
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── polygons
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
|   │   │   │   └── rles
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
```
