## 准备数据集（待更新）

推荐用软链接，将数据集根目录链接到 `$MMSEGMENTATION/data` 里。如果您的文件夹结构是不同的，您也许可以试着修改配置文件里对应的路径。

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   │   ├── VOCaug
│   │   │   ├── dataset
│   │   │   │   ├── cls
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   ├── CHASE_DB1
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── DRIVE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── HRF
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── STARE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
|   ├── dark_zurich
|   │   ├── gps
|   │   │   ├── val
|   │   │   └── val_ref
|   │   ├── gt
|   │   │   └── val
|   │   ├── LICENSE.txt
|   │   ├── lists_file_names
|   │   │   ├── val_filenames.txt
|   │   │   └── val_ref_filenames.txt
|   │   ├── README.md
|   │   └── rgb_anon
|   │   |   ├── val
|   │   |   └── val_ref
|   ├── NighttimeDrivingTest
|   |   ├── gtCoarse_daytime_trainvaltest
|   |   │   └── test
|   |   │       └── night
|   |   └── leftImg8bit
|   |   |   └── test
|   |   |       └── night
│   ├── loveDA
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── potsdam
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── vaihingen
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── iSAID
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── synapse
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── REFUGE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
```

### Cityscapes

注册成功后，数据集可以在 [这里](https://www.cityscapes-dataset.com/downloads/) 下载。

通常情况下，`**labelTrainIds.png` 被用来训练 cityscapes。
基于 [cityscapesscripts](https://github.com/mcordts/cityscapesScripts),
我们提供了一个 [脚本](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py),
去生成 `**labelTrainIds.png`。

```shell
# --nproc 8 意味着有 8 个进程用来转换，它也可以被忽略。
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

### Pascal VOC

Pascal VOC 2012 可以在 [这里](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 下载。
此外，许多最近在 Pascal VOC 数据集上的工作都会利用增广的数据，它们可以在 [这里](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) 找到。

如果您想使用增广后的 VOC 数据集，请运行下面的命令来将数据增广的标注转成正确的格式。

```shell
# --nproc 8 意味着有 8 个进程用来转换，它也可以被忽略。
python tools/convert_datasets/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
```

关于如何拼接数据集 (concatenate) 并一起训练它们，更多细节请参考 [拼接连接数据集](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/tutorials/customize_datasets.md#%E6%8B%BC%E6%8E%A5%E6%95%B0%E6%8D%AE%E9%9B%86) 。

### ADE20K

ADE20K 的训练集和验证集可以在 [这里](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) 下载。
您还可以在 [这里](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip) 下载验证集。

### Pascal Context

Pascal Context 的训练集和验证集可以在 [这里](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar) 下载。
注册成功后，您还可以在 [这里](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar) 下载验证集。

为了从原始数据集里切分训练集和验证集， 您可以在 [这里](https://codalabuser.blob.core.windows.net/public/trainval_merged.json)
下载 trainval_merged.json。

如果您想使用 Pascal Context 数据集，
请安装 [细节](https://github.com/zhanghang1989/detail-api) 然后再运行如下命令来把标注转换成正确的格式。

```shell
python tools/convert_datasets/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
```

### CHASE DB1

CHASE DB1 的训练集和验证集可以在 [这里](https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip) 下载。

为了将 CHASE DB1 数据集转换成 MMSegmentation 的格式，您需要运行如下命令：

```shell
python tools/convert_datasets/chase_db1.py /path/to/CHASEDB1.zip
```

这个脚本将自动生成正确的文件夹结构。

### DRIVE

DRIVE 的训练集和验证集可以在 [这里](https://drive.grand-challenge.org/) 下载。
在此之前，您需要注册一个账号，当前 '1st_manual' 并未被官方提供，因此需要您从其他地方获取。

为了将 DRIVE 数据集转换成 MMSegmentation 格式，您需要运行如下命令：

```shell
python tools/convert_datasets/drive.py /path/to/training.zip /path/to/test.zip
```

这个脚本将自动生成正确的文件夹结构。

### HRF

首先，下载 [healthy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy.zip) [glaucoma.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma.zip), [diabetic_retinopathy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy.zip), [healthy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy_manualsegm.zip), [glaucoma_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma_manualsegm.zip) 以及 [diabetic_retinopathy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy_manualsegm.zip) 。

为了将 HRF 数据集转换成 MMSegmentation 格式，您需要运行如下命令：

```shell
python tools/convert_datasets/hrf.py /path/to/healthy.zip /path/to/healthy_manualsegm.zip /path/to/glaucoma.zip /path/to/glaucoma_manualsegm.zip /path/to/diabetic_retinopathy.zip /path/to/diabetic_retinopathy_manualsegm.zip
```

这个脚本将自动生成正确的文件夹结构。

### STARE

首先，下载 [stare-images.tar](http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar), [labels-ah.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar) 和 [labels-vk.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar) 。

为了将 STARE 数据集转换成 MMSegmentation 格式，您需要运行如下命令：

```shell
python tools/convert_datasets/stare.py /path/to/stare-images.tar /path/to/labels-ah.tar /path/to/labels-vk.tar
```

这个脚本将自动生成正确的文件夹结构。

### Dark Zurich

因为我们只支持在此数据集上测试模型，所以您只需下载[验证集](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip) 。

### Nighttime Driving

因为我们只支持在此数据集上测试模型，所以您只需下载[测试集](http://data.vision.ee.ethz.ch/daid/NighttimeDriving/NighttimeDrivingTest.zip) 。

### LoveDA

可以从 Google Drive 里下载 [LoveDA数据集](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing) 。

或者它还可以从 [zenodo](https://zenodo.org/record/5706578#.YZvN7SYRXdF) 下载, 您需要运行如下命令:

```shell
# Download Train.zip
wget https://zenodo.org/record/5706578/files/Train.zip
# Download Val.zip
wget https://zenodo.org/record/5706578/files/Val.zip
# Download Test.zip
wget https://zenodo.org/record/5706578/files/Test.zip
```

对于 LoveDA 数据集，请运行以下命令下载并重新组织数据集

```shell
python tools/convert_datasets/loveda.py /path/to/loveDA
```

请参照 [这里](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/inference.md) 来使用训练好的模型去预测 LoveDA 测试集并且提交到官网。

关于 LoveDA 的更多细节可以在[这里](https://github.com/Junjue-Wang/LoveDA) 找到。

### ISPRS Potsdam

[Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/)
数据集是一个有着2D 语义分割内容标注的城市遥感数据集。
数据集可以从挑战[主页](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/) 获得。
需要其中的 '2_Ortho_RGB.zip' 和 '5_Labels_all_noBoundary.zip'。

对于 Potsdam 数据集，请运行以下命令下载并重新组织数据集

```shell
python tools/convert_datasets/potsdam.py /path/to/potsdam
```

使用我们默认的配置， 将生成 3456 张图片的训练集和 2016 张图片的验证集。

### ISPRS Vaihingen

[Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/)
数据集是一个有着2D 语义分割内容标注的城市遥感数据集。

数据集可以从挑战 [主页](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
需要其中的 'ISPRS_semantic_labeling_Vaihingen.zip' 和 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip'。

对于 Vaihingen 数据集，请运行以下命令下载并重新组织数据集

```shell
python tools/convert_datasets/vaihingen.py /path/to/vaihingen
```

使用我们默认的配置 (`clip_size`=512, `stride_size`=256)， 将生成 344 张图片的训练集和 398 张图片的验证集。

### iSAID

iSAID 数据集(训练集/验证集/测试集)的图像可以从 [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) 下载.

iSAID 数据集(训练集/验证集)的注释可以从 [iSAID](https://captain-whu.github.io/iSAID/dataset.html) 下载.

该数据集是一个大规模的实例分割(也可以用于语义分割)的遥感数据集.

下载后，在数据集转换前，您需要将数据集文件夹调整成如下格式.

```
│   ├── iSAID
│   │   ├── train
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   │   ├── part2.zip
│   │   │   │   ├── part3.zip
│   │   │   ├── Semantic_masks
│   │   │   │   ├── images.zip
│   │   ├── val
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   ├── Semantic_masks
│   │   │   │   ├── images.zip
│   │   ├── test
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   │   ├── part2.zip
```

```shell
python tools/convert_datasets/isaid.py /path/to/iSAID
```

使用我们默认的配置 (`patch_width`=896, `patch_height`=896,　`overlap_area`=384), 将生成 33,978 张图片的训练集和 11,644 张图片的验证集.

## Synapse dataset

这个数据集可以在这个[网页](https://www.synapse.org/#!Synapse:syn3193805/wiki/) 里被下载.
我们参考了 [TransUNet](https://arxiv.org/abs/2102.04306) 里面的数据集预处理的设置, 它将原始数据集 (30 套 3D 样例) 切分出 18 套用于训练, 12 套用于验证. 请参考以下步骤来准备该数据集:

```shell
unzip RawData.zip
cd ./RawData/Training
```

随后新建 `train.txt` 和 `val.txt`.

根据 TransUNet 来将训练集和验证集如下划分:

train.txt

```none
img0005.nii.gz
img0006.nii.gz
img0007.nii.gz
img0009.nii.gz
img0010.nii.gz
img0021.nii.gz
img0023.nii.gz
img0024.nii.gz
img0026.nii.gz
img0027.nii.gz
img0028.nii.gz
img0030.nii.gz
img0031.nii.gz
img0033.nii.gz
img0034.nii.gz
img0037.nii.gz
img0039.nii.gz
img0040.nii.gz
```

val.txt

```none
img0008.nii.gz
img0022.nii.gz
img0038.nii.gz
img0036.nii.gz
img0032.nii.gz
img0002.nii.gz
img0029.nii.gz
img0003.nii.gz
img0001.nii.gz
img0004.nii.gz
img0025.nii.gz
img0035.nii.gz
```

此时, synapse 数据集包括了以下内容:

```none
├── Training
│   ├── img
│   │   ├── img0001.nii.gz
│   │   ├── img0002.nii.gz
│   │   ├── ...
│   ├── label
│   │   ├── label0001.nii.gz
│   │   ├── label0002.nii.gz
│   │   ├── ...
│   ├── train.txt
│   ├── val.txt
```

随后, 运行下面的数据集转换脚本来处理 synapse 数据集:

```shell
python tools/dataset_converters/synapse.py --dataset-path /path/to/synapse
```

使用我们默认的配置, 将生成 2,211 张 2D 图片的训练集和 1,568 张图片的验证集.

需要注意的是 MMSegmentation 默认的评价指标 (例如平均 Dice 值) 都是基于每帧 2D 图片计算的, 这与基于每套 3D 图片计算评价指标的 [TransUNet](https://arxiv.org/abs/2102.04306) 是不同的.

### REFUGE

在[官网](https://refuge.grand-challenge.org)注册后, 下载 [REFUGE 数据集](https://refuge.grand-challenge.org/REFUGE2Download)  `REFUGE2.zip` , 解压后的内容如下:

```none
├── REFUGE2
│   ├── REFUGE2
│   │   ├── Annotation-Training400.zip
│   │   ├── REFUGE-Test400.zip
│   │   ├── REFUGE-Test-GT.zip
│   │   ├── REFUGE-Training400.zip
│   │   ├── REFUGE-Validation400.zip
│   │   ├── REFUGE-Validation400-GT.zip
│   ├── __MACOSX
```

运行如下命令，就可以按照 REFUGE2018 挑战赛划分数据集的标准将数据集切分成训练集、验证集、测试集:

```shell
python tools/convert_datasets/refuge.py --raw_data_root=/path/to/refuge/REFUGE2/REFUGE2
```

这个脚本将自动生成下面的文件夹结构：

```none
│   ├── REFUGE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
```

其中包括 400 张图片的训练集, 400 张图片的验证集和 400 张图片的测试集.

## Mapillary Vistas Datasets

- Mapillary Vistas 数据集需要在[官方](https://www.mapillary.com/dataset/vistas)注册后下载.
- 假设您将下载的数据集zip文件放于目录 `mmsegmentation/data/mapillary`
- 运行以下命令对数据集进行解压.
  ```bash
  cd data/mapillary
  unzip An-ZjB1Zm61yAZG0ozTymz8I8NqI4x0MrYrh26dq7kPgfu8vf9ImrdaOAVOFYbJ2pNAgUnVGBmbue9lTgdBOb5BbKXIpFs0fpYWqACbrQDChAA2fdX0zS9PcHu7fY8c-FOvyBVxPNYNFQuM.zip
  ```
- 解压后, 您将得到以下结构的Mapillary Vistas Dataset.
  ```none
  mmsegmentation
  ├── mmseg
  ├── tools
  ├── configs
  ├── data
  │   ├── mapillary
  │   │   ├── training
  │   │   │   ├── images
  │   │   │   ├── v1.2
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   └── panoptic
  │   │   │   ├── v2.0
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── panoptic
  |   │   │   │   └── polygons
  │   │   ├── validation
  │   │   │   ├── images
  |   │   │   ├── v1.2
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   └── panoptic
  │   │   │   ├── v2.0
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── panoptic
  |   │   │   │   └── polygons
  ```
- 运行以下命令将RGB三通道标签转换为mask单通道标签
  ```bash
  # --nproc 可选参数, 默认值为 1, 选择是否使用多进程来提高转换速度，推荐使用
  # --version 可选参数, 'v1.2', 'v2.0','all', 默认值为 'all', 选择转化哪一个版本的标签，默认为两个版本都转换
  # 在 'mmsegmentation/' 目录下运行以下命令
  # python tools/dataset_converters/mapillary.py [datasets path] [--nproc 8] [--version all]
  python tools/dataset_converters/mapillary.py data/mapillary --nproc 8 --version all
  ```
  运行结束后, 您将得到以下结构的数据集,转换后的单通道标签将保存到`labels_mask`文件夹下.
* 在数据集配置文件中可通过设置`dataset_type = 'MapillaryDataset_v1_2'`和`dataset_type = 'MapillaryDataset_v2_0'`选择数据集标签版本。
*  查看mapillary数据集配置文件→ [V1.2](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/_base_/datasets/mapillary_v1_2.py) 和  [V2.0](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/_base_/datasets/mapillary_v2_0.py)
  ```none
  mmsegmentation
  ├── mmseg
  ├── tools
  ├── configs
  ├── data
  │   ├── mapillary
  │   │   ├── training
  │   │   │   ├── images
  │   │   │   ├── v1.2
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── labels_mask
  |   │   │   │   └── panoptic
  │   │   │   ├── v2.0
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── labels_mask
  |   │   │   │   ├── panoptic
  |   │   │   │   └── polygons
  │   │   ├── validation
  │   │   │   ├── images
  |   │   │   ├── v1.2
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── labels_mask
  |   │   │   │   └── panoptic
  │   │   │   ├── v2.0
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── labels_mask
  |   │   │   │   ├── panoptic
  |   │   │   │   └── polygons
  ```

* **Mapillary Vistas Datasets标签索引及调色板信息**

* **v1.2 标签信息**

  ```none
  There are 66 labels classes in v1.2
  0--Bird--[165, 42, 42],
  1--Ground Animal--[0, 192, 0],
  2--Curb--[196, 196, 196],
  3--Fence--[190, 153, 153],
  4--Guard Rail--[180, 165, 180],
  5--Barrier--[90, 120, 150],
  6--Wall--[102, 102, 156],
  7--Bike Lane--[128, 64, 255],
  8--Crosswalk - Plain--[140, 140, 200],
  9--Curb Cut--[170, 170, 170],
  10--Parking--[250, 170, 160],
  11--Pedestrian Area--[96, 96, 96],
  12--Rail Track--[230, 150, 140],
  13--Road--[128, 64, 128],
  14--Service Lane--[110, 110, 110],
  15--Sidewalk--[244, 35, 232],
  16--Bridge--[150, 100, 100],
  17--Building--[70, 70, 70],
  18--Tunnel--[150, 120, 90],
  19--Person--[220, 20, 60],
  20--Bicyclist--[255, 0, 0],
  21--Motorcyclist--[255, 0, 100],
  22--Other Rider--[255, 0, 200],
  23--Lane Marking - Crosswalk--[200, 128, 128],
  24--Lane Marking - General--[255, 255, 255],
  25--Mountain--[64, 170, 64],
  26--Sand--[230, 160, 50],
  27--Sky--[70, 130, 180],
  28--Snow--[190, 255, 255],
  29--Terrain--[152, 251, 152],
  30--Vegetation--[107, 142, 35],
  31--Water--[0, 170, 30],
  32--Banner--[255, 255, 128],
  33--Bench--[250, 0, 30],
  34--Bike Rack--[100, 140, 180],
  35--Billboard--[220, 220, 220],
  36--Catch Basin--[220, 128, 128],
  37--CCTV Camera--[222, 40, 40],
  38--Fire Hydrant--[100, 170, 30],
  39--Junction Box--[40, 40, 40],
  40--Mailbox--[33, 33, 33],
  41--Manhole--[100, 128, 160],
  42--Phone Booth--[142, 0, 0],
  43--Pothole--[70, 100, 150],
  44--Street Light--[210, 170, 100],
  45--Pole--[153, 153, 153],
  46--Traffic Sign Frame--[128, 128, 128],
  47--Utility Pole--[0, 0, 80],
  48--Traffic Light--[250, 170, 30],
  49--Traffic Sign (Back)--[192, 192, 192],
  50--Traffic Sign (Front)--[220, 220, 0],
  51--Trash Can--[140, 140, 20],
  52--Bicycle--[119, 11, 32],
  53--Boat--[150, 0, 255],
  54--Bus--[0, 60, 100],
  55--Car--[0, 0, 142],
  56--Caravan--[0, 0, 90],
  57--Motorcycle--[0, 0, 230],
  58--On Rails--[0, 80, 100],
  59--Other Vehicle--[128, 64, 64],
  60--Trailer--[0, 0, 110],
  61--Truck--[0, 0, 70],
  62--Wheeled Slow--[0, 0, 192],
  63--Car Mount--[32, 32, 32],
  64--Ego Vehicle--[120, 10, 10],
  65--Unlabeled--[0, 0, 0]
  ```

  **v2.0 标签信息**

  ```none
  There are 124 labels classes in v2.0
  0--Bird--[165, 42, 42],
  1--Ground Animal--[0, 192, 0],
  2--Ambiguous Barrier--[250, 170, 31],
  3--Concrete Block--[250, 170, 32],
  4--Curb--[196, 196, 196],
  5--Fence--[190, 153, 153],
  6--Guard Rail--[180, 165, 180],
  7--Barrier--[90, 120, 150],
  8--Road Median--[250, 170, 33],
  9--Road Side--[250, 170, 34],
  10--Lane Separator--[128, 128, 128],
  11--Temporary Barrier--[250, 170, 35],
  12--Wall--[102, 102, 156],
  13--Bike Lane--[128, 64, 255],
  14--Crosswalk - Plain--[140, 140, 200],
  15--Curb Cut--[170, 170, 170],
  16--Driveway--[250, 170, 36],
  17--Parking--[250, 170, 160],
  18--Parking Aisle--[250, 170, 37],
  19--Pedestrian Area--[96, 96, 96],
  20--Rail Track--[230, 150, 140],
  21--Road--[128, 64, 128],
  22--Road Shoulder--[110, 110, 110],
  23--Service Lane--[110, 110, 110],
  24--Sidewalk--[244, 35, 232],
  25--Traffic Island--[128, 196, 128],
  26--Bridge--[150, 100, 100],
  27--Building--[70, 70, 70],
  28--Garage--[150, 150, 150],
  29--Tunnel--[150, 120, 90],
  30--Person--[220, 20, 60],
  31--Person Group--[220, 20, 60],
  32--Bicyclist--[255, 0, 0],
  33--Motorcyclist--[255, 0, 100],
  34--Other Rider--[255, 0, 200],
  35--Lane Marking - Dashed Line--[255, 255, 255],
  36--Lane Marking - Straight Line--[255, 255, 255],
  37--Lane Marking - Zigzag Line--[250, 170, 29],
  38--Lane Marking - Ambiguous--[250, 170, 28],
  39--Lane Marking - Arrow (Left)--[250, 170, 26],
  40--Lane Marking - Arrow (Other)--[250, 170, 25],
  41--Lane Marking - Arrow (Right)--[250, 170, 24],
  42--Lane Marking - Arrow (Split Left or Straight)--[250, 170, 22],
  43--Lane Marking - Arrow (Split Right or Straight)--[250, 170, 21],
  44--Lane Marking - Arrow (Straight)--[250, 170, 20],
  45--Lane Marking - Crosswalk--[255, 255, 255],
  46--Lane Marking - Give Way (Row)--[250, 170, 19],
  47--Lane Marking - Give Way (Single)--[250, 170, 18],
  48--Lane Marking - Hatched (Chevron)--[250, 170, 12],
  49--Lane Marking - Hatched (Diagonal)--[250, 170, 11],
  50--Lane Marking - Other--[255, 255, 255],
  51--Lane Marking - Stop Line--[255, 255, 255],
  52--Lane Marking - Symbol (Bicycle)--[250, 170, 16],
  53--Lane Marking - Symbol (Other)--[250, 170, 15],
  54--Lane Marking - Text--[250, 170, 15],
  55--Lane Marking (only) - Dashed Line--[255, 255, 255],
  56--Lane Marking (only) - Crosswalk--[255, 255, 255],
  57--Lane Marking (only) - Other--[255, 255, 255],
  58--Lane Marking (only) - Test--[255, 255, 255],
  59--Mountain--[64, 170, 64],
  60--Sand--[230, 160, 50],
  61--Sky--[70, 130, 180],
  62--Snow--[190, 255, 255],
  63--Terrain--[152, 251, 152],
  64--Vegetation--[107, 142, 35],
  65--Water--[0, 170, 30],
  66--Banner--[255, 255, 128],
  67--Bench--[250, 0, 30],
  68--Bike Rack--[100, 140, 180],
  69--Catch Basin--[220, 128, 128],
  70--CCTV Camera--[222, 40, 40],
  71--Fire Hydrant--[100, 170, 30],
  72--Junction Box--[40, 40, 40],
  73--Mailbox--[33, 33, 33],
  74--Manhole--[100, 128, 160],
  75--Parking Meter--[20, 20, 255],
  76--Phone Booth--[142, 0, 0],
  77--Pothole--[70, 100, 150],
  78--Signage - Advertisement--[250, 171, 30],
  79--Signage - Ambiguous--[250, 172, 30],
  80--Signage - Back--[250, 173, 30],
  81--Signage - Information--[250, 174, 30],
  82--Signage - Other--[250, 175, 30],
  83--Signage - Store--[250, 176, 30],
  84--Street Light--[210, 170, 100],
  85--Pole--[153, 153, 153],
  86--Pole Group--[153, 153, 153],
  87--Traffic Sign Frame--[128, 128, 128],
  88--Utility Pole--[0, 0, 80],
  89--Traffic Cone--[210, 60, 60],
  90--Traffic Light - General (Single)--[250, 170, 30],
  91--Traffic Light - Pedestrians--[250, 170, 30],
  92--Traffic Light - General (Upright)--[250, 170, 30],
  93--Traffic Light - General (Horizontal)--[250, 170, 30],
  94--Traffic Light - Cyclists--[250, 170, 30],
  95--Traffic Light - Other--[250, 170, 30],
  96--Traffic Sign - Ambiguous--[192, 192, 192],
  97--Traffic Sign (Back)--[192, 192, 192],
  98--Traffic Sign - Direction (Back)--[192, 192, 192],
  99--Traffic Sign - Direction (Front)--[220, 220, 0],
  100--Traffic Sign (Front)--[220, 220, 0],
  101--Traffic Sign - Parking--[0, 0, 196],
  102--Traffic Sign - Temporary (Back)--[192, 192, 192],
  103--Traffic Sign - Temporary (Front)--[220, 220, 0],
  104--Trash Can--[140, 140, 20],
  105--Bicycle--[119, 11, 32],
  106--Boat--[150, 0, 255],
  107--Bus--[0, 60, 100],
  108--Car--[0, 0, 142],
  109--Caravan--[0, 0, 90],
  110--Motorcycle--[0, 0, 230],
  111--On Rails--[0, 80, 100],
  112--Other Vehicle--[128, 64, 64],
  113--Trailer--[0, 0, 110],
  114--Truck--[0, 0, 70],
  115--Vehicle Group--[0, 0, 142],
  116--Wheeled Slow--[0, 0, 192],
  117--Water Valve--[170, 170, 170],
  118--Car Mount--[32, 32, 32],
  119--Dynamic--[111, 74, 0],
  120--Ego Vehicle--[120, 10, 10],
  121--Ground--[81, 0, 81],
  122--Static--[111, 111, 0],
  123--Unlabeled--[0, 0, 0]
  ```
