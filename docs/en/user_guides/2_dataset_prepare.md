## Prepare datasets

It is recommended to symlink the dataset root to `$MMSEGMENTATION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

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
```

### Cityscapes

The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration.

By convention, `**labelTrainIds.png` are used for cityscapes training.
We provided a [scripts](https://github.com/open-mmlab/mmsegmentation/blob/1.x/tools/dataset_converters/cityscapes.py) based on [cityscapesscripts](https://github.com/mcordts/cityscapesScripts)
to generate `**labelTrainIds.png`.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/cityscapes.py data/cityscapes --nproc 8
```

### Pascal VOC

Pascal VOC 2012 could be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
Beside, most recent works on Pascal VOC dataset usually exploit extra augmentation data, which could be found [here](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

If you would like to use augmented VOC dataset, please run following command to convert augmentation annotations into proper format.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
```

Please refer to [concat dataset](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/advanced_guides/datasets.md) for details about how to concatenate them and train them together.

### ADE20K

The training and validation set of ADE20K could be download from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
We may also download test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

### Pascal Context

The training and validation set of Pascal Context could be download from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar). You may also download test set from [here](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar) after registration.

To split the training and validation set from original dataset, you may download trainval_merged.json from [here](https://codalabuser.blob.core.windows.net/public/trainval_merged.json).

If you would like to use Pascal Context dataset, please install [Detail](https://github.com/zhanghang1989/detail-api) and then run the following command to convert annotations into proper format.

```shell
python tools/dataset_converters/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
```

### COCO Stuff 10k

The data could be downloaded [here](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip) by wget.

For COCO Stuff 10k dataset, please run the following commands to download and convert the dataset.

```shell
# download
mkdir coco_stuff10k && cd coco_stuff10k
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip

# unzip
unzip cocostuff-10k-v1.1.zip

# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/coco_stuff10k.py /path/to/coco_stuff10k --nproc 8
```

By convention, mask labels in `/path/to/coco_stuff164k/annotations/*2014/*_labelTrainIds.png` are used for COCO Stuff 10k training and testing.

### COCO Stuff 164k

For COCO Stuff 164k dataset, please run the following commands to download and convert the augmented dataset.

```shell
# download
mkdir coco_stuff164k && cd coco_stuff164k
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# unzip
unzip train2017.zip -d images/
unzip val2017.zip -d images/
unzip stuffthingmaps_trainval2017.zip -d annotations/

# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/coco_stuff164k.py /path/to/coco_stuff164k --nproc 8
```

By convention, mask labels in `/path/to/coco_stuff164k/annotations/*2017/*_labelTrainIds.png` are used for COCO Stuff 164k training and testing.

The details of this dataset could be found at [here](https://github.com/nightrome/cocostuff#downloads).

### CHASE DB1

The training and validation set of CHASE DB1 could be download from [here](https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip).

To convert CHASE DB1 dataset to MMSegmentation format, you should run the following command:

```shell
python tools/dataset_converters/chase_db1.py /path/to/CHASEDB1.zip
```

The script will make directory structure automatically.

### DRIVE

The training and validation set of DRIVE could be download from [here](https://drive.grand-challenge.org/). Before that, you should register an account. Currently '1st_manual' is not provided officially.

To convert DRIVE dataset to MMSegmentation format, you should run the following command:

```shell
python tools/dataset_converters/drive.py /path/to/training.zip /path/to/test.zip
```

The script will make directory structure automatically.

### HRF

First, download [healthy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy.zip), [glaucoma.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma.zip), [diabetic_retinopathy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy.zip), [healthy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy_manualsegm.zip), [glaucoma_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma_manualsegm.zip) and [diabetic_retinopathy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy_manualsegm.zip).

To convert HRF dataset to MMSegmentation format, you should run the following command:

```shell
python tools/dataset_converters/hrf.py /path/to/healthy.zip /path/to/healthy_manualsegm.zip /path/to/glaucoma.zip /path/to/glaucoma_manualsegm.zip /path/to/diabetic_retinopathy.zip /path/to/diabetic_retinopathy_manualsegm.zip
```

The script will make directory structure automatically.

### STARE

First, download [stare-images.tar](http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar), [labels-ah.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar) and [labels-vk.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar).

To convert STARE dataset to MMSegmentation format, you should run the following command:

```shell
python tools/dataset_converters/stare.py /path/to/stare-images.tar /path/to/labels-ah.tar /path/to/labels-vk.tar
```

The script will make directory structure automatically.

### Dark Zurich

Since we only support test models on this dataset, you may only download [the validation set](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip).

### Nighttime Driving

Since we only support test models on this dataset, you may only download [the test set](http://data.vision.ee.ethz.ch/daid/NighttimeDriving/NighttimeDrivingTest.zip).

### LoveDA

The data could be downloaded from Google Drive [here](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing).

Or it can be downloaded from [zenodo](https://zenodo.org/record/5706578#.YZvN7SYRXdF), you should run the following command:

```shell
# Download Train.zip
wget https://zenodo.org/record/5706578/files/Train.zip
# Download Val.zip
wget https://zenodo.org/record/5706578/files/Val.zip
# Download Test.zip
wget https://zenodo.org/record/5706578/files/Test.zip
```

For LoveDA dataset, please run the following command to download and re-organize the dataset.

```shell
python tools/dataset_converters/loveda.py /path/to/loveDA
```

Using trained model to predict test set of LoveDA and submit it to server can be found [here](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/user_guides/3_inference.md).

More details about LoveDA can be found [here](https://github.com/Junjue-Wang/LoveDA).

### ISPRS Potsdam

The [Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Potsdam.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
The '2_Ortho_RGB.zip' and '5_Labels_all_noBoundary.zip' are required.

For Potsdam dataset, please run the following command to download and re-organize the dataset.

```shell
python tools/dataset_converters/potsdam.py /path/to/potsdam
```

In our default setting, it will generate 3456 images for training and 2016 images for validation.

### ISPRS Vaihingen

The [Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Vaihingen.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
The 'ISPRS_semantic_labeling_Vaihingen.zip' and 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip' are required.

For Vaihingen dataset, please run the following command to download and re-organize the dataset.

```shell
python tools/dataset_converters/vaihingen.py /path/to/vaihingen
```

In our default setting (`clip_size` =512, `stride_size`=256), it will generate 344 images for training and 398 images for validation.

### iSAID

The data images could be download from [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) (train/val/test)

The data annotations could be download from [iSAID](https://captain-whu.github.io/iSAID/dataset.html) (train/val)

The dataset is a Large-scale Dataset for Instance Segmentation (also have segmantic segmentation) in Aerial Images.

You may need to follow the following structure for dataset preparation after downloading iSAID dataset.

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

In our default setting (`patch_width`=896, `patch_height`=896,　`overlap_area`=384), it will generate 33978 images for training and 11644 images for validation.

## LIP(Look Into Person) dataset

This dataset could be download from [this page](https://lip.sysuhcp.com/overview.php).

Please run the following commands to unzip dataset.

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

The contents of  LIP datasets include:

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

This dataset could be download from [this page](https://www.synapse.org/#!Synapse:syn3193805/wiki/)

To follow the data preparation setting of [TransUNet](https://arxiv.org/abs/2102.04306), which splits original training set (30 scans)
into new training (18 scans) and validation set (12 scans). Please run the following command to prepare the dataset.

```shell
unzip RawData.zip
cd ./RawData/Training
```

Then create `train.txt` and `val.txt` to split dataset.

According to TransUnet, the following is the data set division.

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

The contents of synapse datasets include:

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

Then, use this command to convert synapse dataset.

```shell
python tools/dataset_converters/synapse.py --dataset-path /path/to/synapse
```

Noted that MMSegmentation default evaluation metric (such as mean dice value) is calculated on 2D slice image,
which is not comparable to results of 3D scan in some paper such as [TransUNet](https://arxiv.org/abs/2102.04306).

### REFUGE

Register in [REFUGE Challenge](https://refuge.grand-challenge.org) and download [REFUGE dataset](https://refuge.grand-challenge.org/REFUGE2Download).

Then, unzip `REFUGE2.zip` and the contents of original datasets include:

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

Please run the following command to convert REFUGE dataset:

```shell
python tools/convert_datasets/refuge.py --raw_data_root=/path/to/refuge/REFUGE2/REFUGE2
```

The script will make directory structure below:

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

It includes 400 images for training, 400 images for validation and 400 images for testing which is the same as REFUGE 2018 dataset.

## Mapillary Vistas Datasets

- The dataset could be download [here](https://www.mapillary.com/dataset/vistas) after registration.
- Assumption you have put the dataset zip file in `mmsegmentation/data/mapillary`
- Please run the following commands to unzip dataset.
  ```bash
  cd data/mapillary
  unzip An-ZjB1Zm61yAZG0ozTymz8I8NqI4x0MrYrh26dq7kPgfu8vf9ImrdaOAVOFYbJ2pNAgUnVGBmbue9lTgdBOb5BbKXIpFs0fpYWqACbrQDChAA2fdX0zS9PcHu7fY8c-FOvyBVxPNYNFQuM.zip
  ```
- After unzip, you will get Mapillary Vistas Dataset like this structure.
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
- run following command to convert RGB labels to mask labels.
  ```bash
  # --nproc optional, default 1, whether use multi-progress
  # --version optional, 'v1.2', 'v2.0','all', default 'all', choose convert which version labels
  # run this command at 'mmsegmentation' folder
  # python tools/dataset_converters/mapillary.py [datasets path] [--nproc 8] [--version all]
  python tools/dataset_converters/mapillary.py data/mapillary --nproc 8 --version all
  ```
  After then, you will get this structure, mask labels saved in `labels_mask`.
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

* **View datasets labels index and palette**

* **Mapillary Vistas Datasets labels information**
  **v1.2 information**

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

  **v2.0 information**

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
