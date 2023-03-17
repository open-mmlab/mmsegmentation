# Tutorial 4: Train and test with existing models

MMSegmentation supports training and testing models on a variety of devices, which are described below for single-GPU, distributed, and cluster training and testing, respectively. Through this tutorial, you will learn how to train and test using the scripts provided by MMSegmentation.

## Training and testing on a single GPU

### Training on a single GPU

We provide `tools/train.py` to launch training jobs on a single GPU.
The basic usage is as follows.

```shell
python tools/train.py  ${CONFIG_FILE} [optional arguments]
```

This tool accepts several optional arguments, including:

- `--work-dir ${WORK_DIR}`: Override the working directory.
- `--amp`: Use auto mixed precision training.
- `--resume`: Resume from the latest checkpoint in the work_dir automatically.
- `--cfg-options ${OVERRIDE_CONFIGS}`: Override some settings in the used config, and the key-value pair in xxx=yyy format will be merged into the config file.
  For example, '--cfg-option model.encoder.in_channels=6'. Please see this [guide](./1_config.md#Modify-config-through-script-arguments) for more details.

Below are the optional arguments for the multi-gpu test:

- `--launcher`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `--local_rank`: ID for local rank. If not specified, it will be set to 0.

**Note:** Difference between the argument `--resume` and the field `load_from` in the config file:

`--resume` only determines whether to resume from the latest checkpoint in the work_dir. It is usually used for resuming the training process that is interrupted accidentally.

`load_from` will specify the checkpoint to be loaded and the training iteration starts from 0. It is usually used for fine-tuning.

If you would like to resume training from a specific checkpoint, you can use:

```python
python tools/train.py ${CONFIG_FILE} --resume --cfg-options load_from=${CHECKPOINT}
```

**Training on CPU**: The process of training on the CPU is consistent with single GPU training if a machine does not have GPU. If it has GPUs but not wanting to use them, we just need to disable GPUs before the training process.

```shell
export CUDA_VISIBLE_DEVICES=-1
```

And then run the script [above](#training-on-a-single-gpu).

### Testing on a single GPU

We provide `tools/test.py` to launch training jobs on a single GPU.
The basic usage is as follows.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

This tool accepts several optional arguments, including:

- `--work-dir`: If specified, results will be saved in this directory. If not specified, the results will be automatically saved to `work_dirs/{CONFIG_NAME}`.
- `--show`: Show prediction results at runtime, available when `--show-dir` is not specified.
- `--show-dir`: Directory where painted images will be saved. If specified, the visualized segmentation mask will be saved to the `work_dir/timestamp/show_dir`.
- `--wait-time`: The interval of show (s), which takes effect when `--show` is activated. Default to 2.
- `--cfg-options`:  If specified, the key-value pair in xxx=yyy format will be merged into the config file.
- `--tta`: Test time augmentation option.

**Testing on CPU**: The process of testing on the CPU is consistent with single GPU testing if a machine does not have GPU. If it has GPUs but not wanting to use them, we just need to disable GPUs before the training process.

```shell
export CUDA_VISIBLE_DEVICES=-1
```

then run the script [above](#testing-on-a-single-gpu).

## Training and testing on multiple GPUs and multiple machines

### Training on multiple GPUs

OpenMMLab2.0 implements **distributed** training with `MMDistributedDataParallel`.
We provide `tools/dist_train.sh` to launch training on multiple GPUs.

The basic usage is as follows:

```shell
sh tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments remain the same as stated [above](#training-on-a-single-gpu) and have additional arguments to specify the number of GPUs.

An example:

```shell
# checkpoints and logs saved in WORK_DIR=work_dirs/pspnet_r50-d8_4xb4-80k_ade20k-512x512/
# If work_dir is not set, it will be generated automatically.
sh tools/dist_train.sh configs/pspnet/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py 8 --work-dir work_dirs/pspnet_r50-d8_4xb4-80k_ade20k-512x512
```

**Note**: During training, checkpoints and logs are saved in the same folder structure as the config file under `work_dirs/`. A custom work directory is not recommended since evaluation scripts infer work directories from the config file name. If you want to save your weights somewhere else, please use a symlink, for example:

```shell
ln -s ${YOUR_WORK_DIRS} ${MMSEG}/work_dirs
```

### Testing on multiple GPUs

We provide `tools/dist_test.sh` to launch testing on multiple GPUs.
The basic usage is as follows.

```shell
sh tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments remain the same as stated [above](#testing-on-a-single-gpu) and have additional arguments to specify the number of GPUs.

An example:

```shell
./tools/dist_test.sh configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth 4
```

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs, you need to specify different ports (29500 by default) for each job to avoid communication conflict. Otherwise, there will be an error message saying `RuntimeError: Address already in use`.
If you use `dist_train.sh` to launch training jobs, you can set the port in commands with the environment variable `PORT`.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 sh tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 sh tools/dist_train.sh ${CONFIG_FILE} 4
```

### Training with multiple machines

MMSegmentation relies on `torch.distributed` package for distributed training.
Thus, as a basic usage, one can launch distributed training via PyTorch's [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).

If you launch with multiple machines simply connected with ethernet, you can simply run the following commands:
On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS}
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS}
```

Usually, it is slow if you do not have high-speed networking like InfiniBand.

## Manage jobs with Slurm

[Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.

### Training on a cluster with Slurm

On a cluster managed by Slurm, you can use `slurm_train.sh` to spawn training jobs. It supports both single-node and multi-node training.

The basic usage is as follows:

```shell
[GPUS=${GPUS}] sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} [optional arguments]
```

Below is an example of using 4 GPUs to train PSPNet on a Slurm partition named _dev_, and set the work-dir to some shared file systems.

```shell
GPUS=4 sh tools/slurm_train.sh dev pspnet configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py --work-dir work_dir/pspnet
```

You can check [the source code](../../../tools/slurm_train.sh) to review full arguments and environment variables.

### Testing on a cluster with Slurm

Similar to the training task, MMSegmentation provides `slurm_test.sh` to launch testing jobs.

The basic usage is as follows:

```shell
[GPUS=${GPUS}] sh tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

You can check [the source code](../../../tools/slurm_test.sh) to review full arguments and environment variables.

**Note:** When using Slurm, the port option needs to be set in one of the following ways:

1. Set the port through `--cfg-options`. This is more recommended since it does not change the original configs.

   ```shell
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --cfg-options env_cfg.dist_cfg.port=29500
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --cfg-options env_cfg.dist_cfg.port=29501
   ```

2. Modify the config files to set different communication ports.
   In `config1.py`:

   ```python
   enf_cfg = dict(dist_cfg=dict(backend='nccl', port=29500))
   ```

   In `config2.py`:

   ```python
   enf_cfg = dict(dist_cfg=dict(backend='nccl', port=29501))
   ```

   Then you can launch two jobs with config1.py and config2.py.

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
   ```

3. Set the port in the command using the environment variable 'MASTER_PORT':

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 MASTER_PORT=29500 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 MASTER_PORT=29501 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
```

## Testing and saving segment files

### Basic Usage

When you want to save the results, you can use `--out` to specify the output directory.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${OUTPUT_DIR}
```

Here is an example to save the predicted results from model `fcn_r50-d8_4xb4-80k_ade20k-512x512` on ADE20k validatation dataset.

```shell
python tools/test.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512.py ckpt/fcn_r50-d8_512x512_80k_ade20k_20200614_144016-f8ac5082.pth --out work_dirs/format_results
```

You also can modify the config file to define `output_dir`. We also take
`fcn_r50-d8_4xb4-80k_ade20k-512x512` as example just add
`test_evaluator` in `configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512.py`

```python
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir='work_dirs/format_results')
```

then run command without `--out`:

```shell
python tools/test.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512.py ckpt/fcn_r50-d8_512x512_80k_ade20k_20200614_144016-f8ac5082.pth
```

If you would like to only save the predicted results without evaluation as annotation is not released by the official dataset, you can set `format_only=True` and modify `test_dataloader`.
As there is no annotation in dataset, we remove `dict(type='LoadAnnotations')` from `test_dataloader` Here is the example configuration:

```python
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
    format_only=True,
    output_dir='work_dirs/format_results')
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type = 'ADE20KDataset'
        data_root='data/ade/release_test',
        data_prefix=dict(img_path='testing'),
        # we don't load annotation in test transform pipeline.
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 512), keep_ratio=True),
            dict(type='PackSegInputs')
        ]))
```

then run test command:

```shell
python tools/test.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512.py ckpt/fcn_r50-d8_512x512_80k_ade20k_20200614_144016-f8ac5082.pth
```

### Testing Cityscape dataset and save predicted segment files

We recommend `CityscapesMetric` which is the wrapper of Cityscapes'sdk, when you want to
save the predicted results of Cityscape test dataset to submit them in [Cityscape test server](https://www.cityscapes-dataset.com/submit/). Here is the example configuration:

```python
test_evaluator = dict(
    type='CityscapesMetric',
    format_only=True,
    keep_results=True,
    output_dir='work_dirs/format_results')
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(img_path='leftImg8bit/test'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='PackSegInputs')
        ]))
```

then run test command, for example:

```shell
python tools/test.py configs/fcn/fcn_r18-d8_4xb2-80k_cityscapes-512x1024.py ckpt/fcn_r18-d8_512x1024_80k_cityscapes_20201225_021327-6c50f8b4.pth
```
