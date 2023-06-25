# 教程2：准备数据集

我们建议将数据集根目录符号链接到 `$MMSEGMENTATION/data`。
如果您的目录结构不同，您可能需要更改配置文件中相应的路径。
对于中国境内的用户，我们也推荐通过开源数据平台 [OpenDataLab](https://opendatalab.com/) 来下载dsdl标准数据，以获得更好的下载和使用体验，这里有一个下载dsdl数据集并进行训练的案例[DSDLReadme](../../../configs/dsdl/README.md)，欢迎尝试。

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
│   ├── coco_stuff10k
│   │   ├── images
│   │   │   ├── train2014
│   │   │   ├── test2014
│   │   ├── annotations
│   │   │   ├── train2014
│   │   │   ├── test2014
│   │   ├── imagesLists
│   │   │   ├── train.txt
│   │   │   ├── test.txt
│   │   │   ├── all.txt
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
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

## Cityscapes

Cityscapes [官方网站](https://www.cityscapes-dataset.com/)可以下载 Cityscapes 数据集，按照官网要求注册并登陆后，数据可以在[这里](https://www.cityscapes-dataset.com/downloads/)找到。

按照惯例，`**labelTrainIds.png` 用于 cityscapes 训练。
我们提供了一个基于 [cityscapesscripts](https://github.com/mcordts/cityscapesScripts) 的[脚本](https://github.com/open-mmlab/mmsegmentation/blob/1.x/tools/dataset_converters/cityscapes.py)用于生成 `**labelTrainIds.png`。

```shell
# --nproc 表示 8 个转换进程，也可以省略。
python tools/dataset_converters/cityscapes.py data/cityscapes --nproc 8
```

## Pascal VOC

Pascal VOC 2012 可从[此处](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)下载。
此外，Pascal VOC 数据集的最新工作通常利用额外的增强数据，可以在[这里](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)找到。

如果您想使用增强的 VOC 数据集，请运行以下命令将增强数据的标注转换为正确的格式。

```shell
# --nproc 表示 8 个转换进程，也可以省略。
python tools/dataset_converters/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
```

请参考[拼接数据集文档](../advanced_guides/add_datasets.md#拼接数据集)及 [voc_aug 配置示例](../../../configs/_base_/datasets/pascal_voc12_aug.py)以详细了解如何将它们拼接并合并训练。

## ADE20K

ADE20K 的训练和验证集可以从这个[链接](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)下载。
如果需要下载测试数据集，可以在[官网](http://host.robots.ox.ac.uk/)注册后，下载[测试集](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar)。

## Pascal Context

Pascal Context 的训练和验证集可以从[此处](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar)下载。注册后，您也可以从[此处](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar)下载测试集。

从原始数据集中抽出部分数据作为验证集，您可以从[此处](https://codalabuser.blob.core.windows.net/public/trainval_merged.json)下载 trainval_merged.json 文件。

请先安装 [Detail](https://github.com/zhanghang1989/detail-api) 工具然后运行以下命令将标注转换为正确的格式。

```shell
python tools/dataset_converters/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
```

## COCO Stuff 10k

数据可以通过 wget 在[这里](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip)下载。

对于 COCO Stuff 10k 数据集，请运行以下命令下载并转换数据集。

```shell
# 下载
mkdir coco_stuff10k && cd coco_stuff10k
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip

# 解压
unzip cocostuff-10k-v1.1.zip

# --nproc 表示 8 个转换进程，也可以省略。
python tools/dataset_converters/coco_stuff10k.py /path/to/coco_stuff10k --nproc 8
```

按照惯例，`/path/to/coco_stuff164k/annotations/*2014/*_labelTrainIds.png` 中的 mask 标注用于 COCO Stuff 10k 的训练和测试。

## COCO Stuff 164k

对于 COCO Stuff 164k 数据集，请运行以下命令下载并转换增强的数据集。

```shell
# 下载
mkdir coco_stuff164k && cd coco_stuff164k
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# 解压
unzip train2017.zip -d images/
unzip val2017.zip -d images/
unzip stuffthingmaps_trainval2017.zip -d annotations/

# --nproc 表示 8 个转换进程，也可以省略。
python tools/dataset_converters/coco_stuff164k.py /path/to/coco_stuff164k --nproc 8
```

按照惯例，`/path/to/coco_stuff164k/annotations/*2017/*_labelTrainIds.png` 中的 mask 标注用于 COCO Stuff 164k 的训练和测试。

此数据集的详细信息可在[此处](https://github.com/nightrome/cocostuff#downloads)找到。

## CHASE DB1

CHASE DB1 的训练和验证集可以从[此处](https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip)下载。

请运行以下命令，准备 CHASE DB1 数据集：

```shell
python tools/dataset_converters/chase_db1.py /path/to/CHASEDB1.zip
```

该脚本将自动调整数据集目录结构，使其满足 MMSegmentation 数据集加载要求。

## DRIVE

按照[官网](https://drive.grand-challenge.org/)要求，注册并登陆后，便可以下载 DRIVE 的训练和验证数据集。

要将 DRIVE 数据集转换为 MMSegmentation 的格式，请运行以下命令：

```shell
python tools/dataset_converters/drive.py /path/to/training.zip /path/to/test.zip
```

该脚本将自动调整数据集目录结构，使其满足 MMSegmentation 数据集加载要求。

## HRF

请下载 [health.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy.zip)、[glaucoma.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma.zip)、[diabetic_retinopathy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy.zip)、[healthy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy_manualsegm.zip)、[glaucoma_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma_manualsegm.zip) 和 [diabetic_retinopathy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy_manualsegm.zip)，无需解压，可以直接运行以下命令，准备 HRF 数据集：

```shell
python tools/dataset_converters/hrf.py /path/to/healthy.zip /path/to/healthy_manualsegm.zip /path/to/glaucoma.zip /path/to/glaucoma_manualsegm.zip /path/to/diabetic_retinopathy.zip /path/to/diabetic_retinopathy_manualsegm.zip
```

该脚本将自动调整数据集目录结构，使其满足 MMSegmentation 数据集加载要求。

## STARE

请下载 [stare images.tar](http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar)、[labels-ah.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar) 和 [labels-vk.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar)，无需解压，可以直接运行以下命令，准备 STARE 数据集：

```shell
python tools/dataset_converters/stare.py /path/to/stare-images.tar /path/to/labels-ah.tar /path/to/labels-vk.tar
```

该脚本将自动调整数据集目录结构，使其满足 MMSegmentation 数据集加载要求。

## Dark Zurich

由于我们只支持在此数据集上的模型测试，因此您只需要下载并解压[验证数据集](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip)。

## Nighttime Driving

由于我们只支持在此数据集上的模型测试，因此您只需要下载并解压[验证数据集](http://data.vision.ee.ethz.ch/daid/NighttimeDriving/NighttimeDrivingTest.zip)。

## LoveDA

数据可以从[此处](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing)下载 LaveDA 数据集。

或者可以从 [zenodo](https://zenodo.org/record/5706578#.YZvN7SYRXdF) 下载。下载后，无需解压，直接运行以下命令：

```shell
# 下载 Train.zip
wget https://zenodo.org/record/5706578/files/Train.zip
# 下载 Val.zip
wget https://zenodo.org/record/5706578/files/Val.zip
# 下载 Test.zip
wget https://zenodo.org/record/5706578/files/Test.zip
```

请对于 LoveDA 数据集，请运行以下命令调整数据集目录。

```shell
python tools/dataset_converters/loveda.py /path/to/loveDA
```

可将模型对 LoveDA 的测试集的预测结果上传至到数据集[测试服务器](https://codalab.lisn.upsaclay.fr/competitions/421)，查看评测结果。

有关 LoveDA 的更多详细信息，可查看[此处](https://github.com/Junjue-Wang/LoveDA).

## ISPRS Potsdam

[Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) 城市语义分割数据集用于 2D 语义分割竞赛 —— Potsdam。

数据集可以在竞赛[主页](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)上请求获得。
这里也提供了[BaiduNetdisk](https://pan.baidu.com/s/1K-cLVZnd1X7d8c26FQ-nGg?pwd=mseg)，提取码：mseg、 [Google Drive](https://drive.google.com/drive/folders/1w3EJuyUGet6_qmLwGAWZ9vw5ogeG0zLz?usp=sharing)以及[OpenDataLab](https://opendatalab.com/ISPRS_Potsdam/download)。
实验中需要下载 '2_Ortho_RGB.zip' 和 '5_Labels_all_noBoundary.zip'。

对于 Potsdam 数据集，请运行以下命令调整数据集目录。

```shell
python tools/dataset_converters/potsdam.py /path/to/potsdam
```

在我们的默认设置中，将生成 3456 张图像用于训练和 2016 张图像用于验证。

## ISPRS Vaihingen

[Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx) 城市语义分割数据集用于 2D 语义分割竞赛 —— Vaihingen。

数据集可以在竞赛[主页](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)上请求获得。
这里也提供了[BaiduNetdisk](https://pan.baidu.com/s/109D3WLrLafsuYtLeerLiiA?pwd=mseg)，提取码：mseg 、 [Google Drive](https://drive.google.com/drive/folders/1w3NhvLVA2myVZqOn2pbiDXngNC7NTP_t?usp=sharing)。
实验中需要下载 'ISPRS_semantic_labeling_Vaihingen.zip' 和 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip'。

对于 Vaihingen 数据集，请运行以下命令调整数据集目录。

```shell
python tools/dataset_converters/vaihingen.py /path/to/vaihingen
```

在我们的默认设置（`clip_size`=512, `stride_size`=256）中，将生成 344 张图像用于训练和 398 张图像用于验证。

## iSAID

iSAID 数据集可从 [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) 下载训练/验证/测试数据集的图像数据，

并从 [iSAID](https://captain-whu.github.io/iSAID/dataset.html)下载训练/验证数据集的标注数据。

该数据集是航空图像实例分割和语义分割任务的大规模数据集。

下载 iSAID 数据集后，您可能需要按照以下结构进行数据集准备。

```none
├── data
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
python tools/dataset_converters/isaid.py /path/to/iSAID
```

在我们的默认设置（`patch_width`=896, `patch_height`=896, `overlap_area`=384）中，将生成 33978 张图像用于训练和 11644 张图像用于验证。

## LIP(Look Into Person) dataset

该数据集可以从[此页面](https://lip.sysuhcp.com/overview.php)下载。

请运行以下命令来解压数据集。

```shell
unzip LIP.zip
cd LIP
unzip TrainVal_images.zip
unzip TrainVal_parsing_annotations.zip
cd TrainVal_parsing_annotations
unzip TrainVal_parsing_annotations.zip
mv train_segmentations ../
mv val_segmentations ../
cd ..
```

LIP 数据集的内容包括：

```none
├── data
│   ├── LIP
│   │   ├── train_images
│   │   │   ├── 1000_1234574.jpg
│   │   │   ├── ...
│   │   ├── train_segmentations
│   │   │   ├── 1000_1234574.png
│   │   │   ├── ...
│   │   ├── val_images
│   │   │   ├── 100034_483681.jpg
│   │   │   ├── ...
│   │   ├── val_segmentations
│   │   │   ├── 100034_483681.png
│   │   │   ├── ...
```

## Synapse dataset

此数据集可以从[此页面](https://www.synapse.org/#!Synapse:syn3193805/wiki/)下载。

遵循 [TransUNet](https://arxiv.org/abs/2102.04306) 的数据准备设定，将原始训练集（30 次扫描）拆分为新的训练集（18 次扫描）和验证集（12 次扫描）。请运行以下命令来准备数据集。

```shell
unzip RawData.zip
cd ./RawData/Training
```

然后创建 `train.txt` 和 `val.txt` 以拆分数据集。

根据 TransUnet，以下是数据集的划分。

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

synapse 数据集的内容包括：

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

然后，使用此命令转换 synapse 数据集。

```shell
python tools/dataset_converters/synapse.py --dataset-path /path/to/synapse
```

注意，MMSegmentation 的默认评估指标（例如 mean dice value）是在 2D 切片图像上计算的，这与 [TransUNet](https://arxiv.org/abs/2102.04306) 等一些论文中的 3D 扫描结果是不同的。

## REFUGE

在 [REFUGE Challenge](https://refuge.grand-challenge.org) 官网上注册并下载 [REFUGE 数据集](https://refuge.grand-challenge.org/REFUGE2Download)。

然后，解压 `REFUGE2.zip`，原始数据集的内容包括：

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

请运行以下命令转换 REFUGE 数据集：

```shell
python tools/convert_datasets/refuge.py --raw_data_root=/path/to/refuge/REFUGE2/REFUGE2
```

脚本会将目录结构转换如下：

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

包含 400 张用于训练的图像、400 张用于验证的图像和 400 张用于测试的图像，这与 REFUGE 2018 数据集相同。

## Mapillary Vistas Datasets

- Mapillary Vistas [官方网站](https://www.mapillary.com/dataset/vistas) 可以下载 Mapillary Vistas 数据集，按照官网要求注册并登陆后，数据可以在[这里](https://www.mapillary.com/dataset/vistas)找到。

- Mapillary Vistas 数据集使用 8-bit with color-palette 来存储标签。不需要进行转换操作。

- 假设您已将数据集 zip 文件放在 `mmsegmentation/data/mapillary` 中

- 请运行以下命令来解压数据集。

  ```bash
  cd data/mapillary
  unzip An-ZjB1Zm61yAZG0ozTymz8I8NqI4x0MrYrh26dq7kPgfu8vf9ImrdaOAVOFYbJ2pNAgUnVGBmbue9lTgdBOb5BbKXIpFs0fpYWqACbrQDChAA2fdX0zS9PcHu7fY8c-FOvyBVxPNYNFQuM.zip
  ```

- 解压后，您将获得类似于此结构的 Mapillary Vistas 数据集。语义分割 mask 标签在 `labels` 文件夹中。

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

- 您可以在配置中使用 `MapillaryDataset_v1` 和 `Mapillary Dataset_v2` 设置数据集版本。
  在此处 [V1.2](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/datasets/mapillary_v1.py) 和 [V2.0](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/datasets/mapillary_v2.py) 查看 Mapillary Vistas 数据集配置文件

## LEVIR-CD

[LEVIR-CD](https://justchenhao.github.io/LEVIR/) 大规模遥感建筑变化检测数据集。

数据集可以在[主页](https://justchenhao.github.io/LEVIR/)上请求获得。

数据集的补充版本可以在[主页](https://github.com/S2Looking/Dataset)上请求获得。

请下载数据集的补充版本，然后解压 `LEVIR-CD+.zip`，数据集的内容包括：

```none
│   ├── LEVIR-CD+
│   │   ├── train
│   │   │   ├── A
│   │   │   ├── B
│   │   │   ├── label
│   │   ├── test
│   │   │   ├── A
│   │   │   ├── B
│   │   │   ├── label
```

对于 LEVIR-CD 数据集，请运行以下命令无重叠裁剪影像：

```shell
python tools/dataset_converters/levircd.py --dataset-path /path/to/LEVIR-CD+ --out_dir /path/to/LEVIR-CD
```

裁剪后的影像大小为256x256，与原论文保持一致。
