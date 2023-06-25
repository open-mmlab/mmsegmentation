# DSDL: Standard Description Language for DataSet

<!-- [ALGORITHM] -->

<!-- [DATASET] -->

## Abstract

<!-- [ABSTRACT] -->

Data is the cornerstone of artificial intelligence. The efficiency of data acquisition, exchange, and application directly impacts the advances in technologies and applications. Over the long history of AI, a vast quantity of data sets have been developed and distributed. However, these datasets are defined in very different forms, which incurs significant overhead when it comes to exchange, integration, and utilization -- it is often the case that one needs to develop a new customized tool or script in order to incorporate a new dataset into a workflow.

To overcome such difficulties, we develop **Data Set Description Language (DSDL)**. More details please visit our [official documents](https://opendatalab.github.io/dsdl-docs/getting_started/overview/), dsdl datasets can be downloaded from our platform [OpenDataLab](https://opendatalab.com/).

<!-- [IMAGE] -->

## Steps

- install dsdl and opendatalab:

  ```
  pip install dsdl
  pip install opendatalab
  ```

- install mmseg and pytorch:
  please refer this [installation documents](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

- prepare dsdl dataset (take voc2012 as an example)

  - dowaload dsdl dataset (you will need an opendatalab account to do so. [register one now](https://opendatalab.com/))

    ```
    cd data

    odl login
    odl get PASCAL_VOC2012
    ```

    usually, dataset is compressed on opendatalab platform, the downloaded voc 2012 dataset should be like this:

    ```
    data/
    ├── PASCAL_VOC2012
    │   ├── dsdl
    │   │   ├── dsdl_Det_full.zip
    │   │   └── dsdl_SemSeg_full.zip
    │   ├── raw
    │   │   ├── VOC2012test.tar
    │   │   ├── VOCdevkit_18-May-2011.tar
    │   │   └── VOCtrainval_11-May-2012.tar
    │   └── README.md
    └── ...
    ```

  - decompress dataset

    ```
    cd dsdl
    unzip dsdl_SemSeg_full.zip
    ```

    as we do not need detection dsdl files, we only decompress the semantic segmentation files here.

    ```
    cd ../raw
    tar -xvf VOCtrainval_11-May-2012.tar
    tar -xvf VOC2012test.tar

    cd ../../
    ```

- change traning config

  open the [voc config file](voc.py) and set some file paths as below:

  ```
  data_root = 'data/PASCAL_VOC2012'
  img_prefix = 'raw/VOCdevkit/VOC2012'
  train_ann = 'dsdl/dsdl_SemSeg_full/set-train/train.yaml'
  val_ann = 'dsdl/dsdl_SemSeg_full/set-val/val.yaml'
  ```

  as dsdl datasets with one task using one dataloader, we can simplly change these file paths to train a model on a different dataset.

- train:

  - using single gpu:

  ```
  python tools/train.py {config_file}
  ```

  - using slrum:

  ```
  ./tools/slurm_train.sh {partition} {job_name} {config_file} {work_dir} {gpu_nums}
  ```

## Test Results

|  Datasets  |                                                                                        Model                                                                                         | mIoU(%) |          Config           |
| :--------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----: | :-----------------------: |
|  voc2012   |    [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug/deeplabv3_r50-d8_512x512_20k_voc12aug_20200617_010906-596905ef.pth)    |  76.73  |    [config](./voc.py)     |
| cityscapes | [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth) |  79.01  | [config](./cityscapes.py) |
