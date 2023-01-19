## Prepare datasets

It is recommended to symlink the dataset root to `$MMSEGMENTATION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

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

## Mapillary Vistas Datasets

- The dataset could be download [here](https://www.mapillary.com/dataset/vistas) after registration.
- Assumption you have put the dataset zip file in `mmsegmentation/data`
- Please run the following commands to unzip dataset.
  ```bash
  cd data
  mkdir mapillary
  unzip -d mapillary An-ZjB1Zm61yAZG0ozTymz8I8NqI4x0MrYrh26dq7kPgfu8vf9ImrdaOAVOFYbJ2pNAgUnVGBmbue9lTgdBOb5BbKXIpFs0fpYWqACbrQDChAA2fdX0zS9PcHu7fY8c-FOvyBVxPNYNFQuM.zip
  ```
- After unzip, you will get Mapillary Vistas Dataset like this structure.
  ```none
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
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   └── panoptic
  │   │   │   ├── v2.0
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── panoptic
  |   │   │   │   └── polygons
  ```
- run following commands to convert RGB labels to mask labels
  ```bash
  # --nproc optional, default 1, whether use multi-progress
  # --version optional, 'v1.2', 'v2.0','all', default 'all', choose convert which version labels
  # run this command at 'mmsegmentation/projects/Mapillary_dataset' folder
  cd mmsegmentation/projects/mapillary_dataset
  python tools/dataset_converters/mapillary.py ../../data/mapillary --nproc 8 --version all
  ```
  After then, you will get this structure
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
