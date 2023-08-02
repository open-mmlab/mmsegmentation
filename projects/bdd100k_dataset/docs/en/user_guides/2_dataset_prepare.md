## BDD100K

- You could download BDD100k datasets from  [here](https://bdd-data.berkeley.edu/) after  registration.

- You can download images and masks by clicking  `10K Images` button and `Segmentation` button.

- After download, unzip by the following instructions:

  ```bash
  unzip ~/bdd100k_images_10k.zip -d ~/mmsegmentation/data/
  unzip ~/bdd100k_sem_seg_labels_trainval.zip -d ~/mmsegmentation/data/
  ```

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
