## 准备数据集

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

关于如何拼接数据集 (concatenate) 并一起训练它们，更多细节请参考 [拼接连接数据集](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/tutorials/new_dataset.md#concatenate-dataset) 。

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
