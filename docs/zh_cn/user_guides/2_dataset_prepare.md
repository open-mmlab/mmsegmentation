## 准备数据集（待更新）

推荐用软链接, 将数据集根目录链接到 `$MMSEGMENTATION/data` 里. 如果您的文件夹结构是不同的, 您也许可以试着修改配置文件里对应的路径.

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
```

### Cityscapes

注册成功后, 数据集可以在 [这里](https://www.cityscapes-dataset.com/downloads/) 下载.

通常情况下, `**labelTrainIds.png` 被用来训练 cityscapes.
基于 [cityscapesscripts](https://github.com/mcordts/cityscapesScripts),
我们提供了一个 [脚本](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py),
去生成 `**labelTrainIds.png`.

```shell
# --nproc 8 意味着有 8 个进程用来转换,它也可以被忽略.
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

### Pascal VOC

Pascal VOC 2012 可以在 [这里](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 下载.
此外, 许多最近在 Pascal VOC 数据集上的工作都会利用增广的数据, 它们可以在 [这里](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) 找到.

如果您想使用增广后的 VOC 数据集, 请运行下面的命令来将数据增广的标注转成正确的格式.

```shell
# --nproc 8 意味着有 8 个进程用来转换,它也可以被忽略.
python tools/convert_datasets/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
```

关于如何拼接数据集 (concatenate) 并一起训练它们, 更多细节请参考 [拼接连接数据集](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/tutorials/customize_datasets.md#%E6%8B%BC%E6%8E%A5%E6%95%B0%E6%8D%AE%E9%9B%86) .

### ADE20K

ADE20K 的训练集和验证集可以在 [这里](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) 下载.
您还可以在 [这里](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip) 下载验证集.

### Pascal Context

Pascal Context 的训练集和验证集可以在 [这里](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar) 下载.
注册成功后, 您还可以在 [这里](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar) 下载验证集.

为了从原始数据集里切分训练集和验证集, 您可以在 [这里](https://codalabuser.blob.core.windows.net/public/trainval_merged.json)
下载 trainval_merged.json.

如果您想使用 Pascal Context 数据集,
请安装 [细节](https://github.com/zhanghang1989/detail-api) 然后再运行如下命令来把标注转换成正确的格式.

```shell
python tools/convert_datasets/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
```

### CHASE DB1

CHASE DB1 的训练集和验证集可以在 [这里](https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip) 下载.

为了将 CHASE DB1 数据集转换成 MMSegmentation 的格式,您需要运行如下命令:

```shell
python tools/convert_datasets/chase_db1.py /path/to/CHASEDB1.zip
```

这个脚本将自动生成正确的文件夹结构.

### DRIVE

DRIVE 的训练集和验证集可以在 [这里](https://drive.grand-challenge.org/) 下载.
在此之前, 您需要注册一个账号, 当前 '1st_manual' 并未被官方提供, 因此需要您从其他地方获取.

为了将 DRIVE 数据集转换成 MMSegmentation 格式, 您需要运行如下命令:

```shell
python tools/convert_datasets/drive.py /path/to/training.zip /path/to/test.zip
```

这个脚本将自动生成正确的文件夹结构.

### HRF

首先, 下载 [healthy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy.zip) [glaucoma.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma.zip), [diabetic_retinopathy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy.zip), [healthy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy_manualsegm.zip), [glaucoma_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma_manualsegm.zip) 以及 [diabetic_retinopathy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy_manualsegm.zip).

为了将 HRF 数据集转换成 MMSegmentation 格式, 您需要运行如下命令:

```shell
python tools/convert_datasets/hrf.py /path/to/healthy.zip /path/to/healthy_manualsegm.zip /path/to/glaucoma.zip /path/to/glaucoma_manualsegm.zip /path/to/diabetic_retinopathy.zip /path/to/diabetic_retinopathy_manualsegm.zip
```

这个脚本将自动生成正确的文件夹结构.

### STARE

首先, 下载 [stare-images.tar](http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar), [labels-ah.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar) 和 [labels-vk.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar).

为了将 STARE 数据集转换成 MMSegmentation 格式, 您需要运行如下命令:

```shell
python tools/convert_datasets/stare.py /path/to/stare-images.tar /path/to/labels-ah.tar /path/to/labels-vk.tar
```

这个脚本将自动生成正确的文件夹结构.

### Dark Zurich

因为我们只支持在此数据集上测试模型, 所以您只需下载[验证集](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip).

### Nighttime Driving

因为我们只支持在此数据集上测试模型,所以您只需下载[测试集](http://data.vision.ee.ethz.ch/daid/NighttimeDriving/NighttimeDrivingTest.zip).

### LoveDA

可以从 Google Drive 里下载 [LoveDA数据集](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing).

或者它还可以从 [zenodo](https://zenodo.org/record/5706578#.YZvN7SYRXdF) 下载, 您需要运行如下命令:

```shell
# Download Train.zip
wget https://zenodo.org/record/5706578/files/Train.zip
# Download Val.zip
wget https://zenodo.org/record/5706578/files/Val.zip
# Download Test.zip
wget https://zenodo.org/record/5706578/files/Test.zip
```

对于 LoveDA 数据集,请运行以下命令下载并重新组织数据集:

```shell
python tools/convert_datasets/loveda.py /path/to/loveDA
```

请参照 [这里](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/inference.md) 来使用训练好的模型去预测 LoveDA 测试集并且提交到官网.

关于 LoveDA 的更多细节可以在[这里](https://github.com/Junjue-Wang/LoveDA) 找到.

### ISPRS Potsdam

[Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/)
数据集是一个有着2D 语义分割内容标注的城市遥感数据集.
数据集可以从挑战[主页](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/) 获得.
需要其中的 `2_Ortho_RGB.zip` 和 `5_Labels_all_noBoundary.zip`.

对于 Potsdam 数据集,请运行以下命令下载并重新组织数据集

```shell
python tools/convert_datasets/potsdam.py /path/to/potsdam
```

使用我们默认的配置, 将生成 3,456 张图片的训练集和 2,016 张图片的验证集.

### ISPRS Vaihingen

[Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/)
数据集是一个有着2D 语义分割内容标注的城市遥感数据集.

数据集可以从挑战 [主页](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
需要其中的 'ISPRS_semantic_labeling_Vaihingen.zip' 和 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip'.

对于 Vaihingen 数据集, 请运行以下命令下载并重新组织数据集

```shell
python tools/convert_datasets/vaihingen.py /path/to/vaihingen
```

使用我们默认的配置 (`clip_size`=512, `stride_size`=256), 将生成 344 张图片的训练集和 398 张图片的验证集.

### iSAID

iSAID 数据集(训练集/验证集/测试集)的图像可以从 [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) 下载.

iSAID 数据集(训练集/验证集)的注释可以从 [iSAID](https://captain-whu.github.io/iSAID/dataset.html) 下载.

该数据集是一个大规模的实例分割(也可以用于语义分割)的遥感数据集.

下载后, 在数据集转换前, 您需要将数据集文件夹调整成如下格式.

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
