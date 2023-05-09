# KITTI STEP Dataset

Support **`KITTI STEP Dataset`**

## Description

Author: TimoK93

This project implements **`KITTI STEP Dataset`**

### Dataset preparing

After registration, the data images could be download from [KITTI-STEP](http://www.cvlibs.net/datasets/kitti/eval_step.php)

You may need to follow the following structure for dataset preparation after downloading KITTI-STEP dataset.

```
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── kitti_step
│   │   ├── testing
│   │   ├── training
│   │   ├── panoptic_maps
```

Run the preparation script to generate label files and kitti subsets by executing

```shell
python tools/convert_datasets/kitti_step.py /path/to/kitti_step
```

After executing the script, your directory should look like

```
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── kitti_step
│   │   ├── testing
│   │   ├── training
│   │   ├── panoptic_maps
│   │   ├── training_openmmlab
│   │   ├── panoptic_maps_openmmlab
```

### Training commands

```bash
# Dataset train commands
# at `mmsegmentation` folder
bash tools/dist_train.sh projects/kitti_step_dataset/configs/segformer/segformer_mit-b5_368x368_160k_kittistep.py 8
```

### Testing commands

```bash
mim test mmsegmentation projects/kitti_step_dataset/configs/segformer/segformer_mit-b5_368x368_160k_kittistep.py --work-dir work_dirs/segformer_mit-b5_368x368_160k_kittistep --checkpoint ${CHECKPOINT_PATH} --eval mIoU
```

| Method    | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                 | model                                                                                                                                                                              | log                                                                                                                                                                          |
| --------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Segformer | MIT-B5   | 368x368   |  160000 | -        | -              | 65.05 |             - | [config](configs/segformer/segformer_mit-b5_368x368_160k_kittistep.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_368x368_160k_kittistep/segformer_mit-b5_368x368_160k_kittistep_20230506_103002-20797496.pth) | [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_368x368_160k_kittistep/segformer_mit-b5_368x368_160k_kittistep_20230506_103002.log.json) |

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

  - [x] Basic docstrings & proper citation

  - [ ] Test-time correctness

  - [x] A full README

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

  - [ ] Unit tests

  - [ ] Code polishing

  - [ ] Metafile.yml

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
