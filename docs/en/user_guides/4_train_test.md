# Tutorial 4: Train and test with existing models

This tutorial provides instruction for users to use the models provided in the [Model Zoo](../model_zoo.md) for other datasets to obtain better performance.
MMSegmentation also provides out-of-the-box tools for training models.
This section will show how to train and test models on standard datasets.

## Train models on standard datasets

### Modify training schedule

Modify the following configuration to customize the training.

```python
# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
# basic hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'))
```

### Use pre-trained model

Users can load a pre-trained model by setting the `load_from` field of the config to the model's path or link.
The users might need to download the model weights before training to avoid the download time during training.

```python
# use the pre-trained model for the whole PSPNet
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'  # model path can be found in model zoo
```

### Training on a single GPU

We provide `tools/train.py` to launch training jobs on a single GPU.
The basic usage is as follows.
This tool accepts several optional arguments, including:

- `--work-dir ${WORK_DIR}`: Override the working directory.
- `--amp`: Use auto mixed precision training.
- `--resume ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--cfg-options ${OVERRIDE_CONFIGS}`: Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.
  For example, '--cfg-option model.encoder.in_channels=6'. Please see this [guide](./1_config.md#Modify-config-through-script-arguments) for more details.
  Below is the optional arguments for multi-gpu test:
- `--launcher`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `--local_rank`: ID for local rank. If not specified, it will be set to 0.
  **Note**:
  Difference between `--resume` and `load-from`:
  `--resume` loads both the model weights and optimizer status, and the iteration is also inherited from the specified checkpoint.
  It is usually used for resuming the training process that is interrupted accidentally.

`load-from` only loads the model weights and the training iteration starts from 0. It is usually used for fine-tuning.

### Training on CPU

The process of training on the CPU is consistent with single GPU training if machine does not have GPU. If it has GPUs but not wanting to use it, we just need to disable GPUs before the training process.

```shell
export CUDA_VISIBLE_DEVICES=-1
```

And then run the script [above](#training-on-a-single-GPU).

```{warning}
The process of training on the CPU is consistent with single GPU training. We just need to disable GPUs before the training process.
```

### Training on multiple GPUs

MMSegmentation implements **distributed** training with `MMDistributedDataParallel`.
We provide `tools/dist_train.sh` to launch training on multiple GPUs.
The basic usage is as follows.

```shell
sh tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

Optional arguments remain the same as stated [above](#training-on-a-single-gpu)
and has additional arguments to specify the number of GPUs.
An example:

```shell
# checkpoints and logs saved in WORK_DIR=work_dirs/pspnet_r50-d8_512x512_80k_ade20k/
# If work_dir is not set, it will be generated automatically.
sh tools/dist_train.sh configs/pspnet/pspnet_r50-d8_512x512_80k_ade20k.py 8 --work-dir work_dirs/pspnet_r50-d8_512x512_80k_ade20k
```

**Note**: During training, checkpoints and logs are saved in the same folder structure as the config file under `work_dirs/`. Custom work directory is not recommended since evaluation scripts infer work directories from the config file name. If you want to save your weights somewhere else, please use symlink, for example:

```shell
ln -s ${YOUR_WORK_DIRS} ${MMSEG}/work_dirs
```

#### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs, you need to specify different ports (29500 by default) for each job to avoid communication conflict. Otherwise, there will be error message saying `RuntimeError: Address already in use`.
If you use `dist_train.sh` to launch training jobs, you can set the port in commands with environment variable `PORT`.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 sh tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 sh tools/dist_train.sh ${CONFIG_FILE} 4
```

### Training on multiple nodes

MMSegmentation relies on `torch.distributed` package for distributed training.
Thus, as a basic usage, one can launch distributed training via PyTorch's [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).

#### Train with multiple machines

If you launch with multiple machines simply connected with ethernet, you can simply run following commands:
On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS}
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS}
```

Usually it is slow if you do not have high speed networking like InfiniBand.

#### Manage jobs with Slurm

[Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use `slurm_train.sh` to spawn training jobs. It supports both single-node and multi-node training.
The basic usage is as follows.

```shell
[GPUS=${GPUS}] sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

Below is an example of using 4 GPUs to train PSPNet on a Slurm partition named _dev_, and set the work-dir to some shared file systems.

```shell
GPUS=4 sh tools/slurm_train.sh dev pspnet configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py --work-dir work_dir/pspnet
```

You can check [the source code](../../../tools/dist_train.sh) to review full arguments and environment variables.
When using Slurm, the port option need to be set in one of the following ways:

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

## Test models on standard datasets

We provide testing scripts for evaluating an existing model on the whole dataset.
The following testing environments are supported:

- single GPU
- CPU
- single node multiple GPU
- multiple node

Choose the proper script to perform testing depending on the testing environment.

```shell
# single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--work-dir ${WORK_DIR}] \
    [--show ${SHOW_RESULTS}] \
    [--show-dir ${VISUALIZATION_DIRECTORY}] \
    [--wait-time ${SHOW_INTERVAL}] \
    [--cfg-options ${OVERRIDE_CONFIGS}]
# CPU testing
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--work-dir ${WORK_DIR}] \
    [--show ${SHOW_RESULTS}] \
    [--show-dir ${VISUALIZATION_DIRECTORY}] \
    [--wait-time ${SHOW_INTERVAL}] \
    [--cfg-options ${OVERRIDE_CONFIGS}]
# multi-gpu testing
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--work-dir ${WORK_DIR}] \
    [--cfg-options ${OVERRIDE_CONFIGS}]
```

`tools/dist_test.sh` also supports multi-node testing, but relies on PyTorch's [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).
[Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use `slurm_test.sh` to spawn testing jobs. It supports both single-node and multi-node testing.

```shell
[GPUS=${GPUS}] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} \
    ${CONFIG_FILE} ${CHECKPOINT_FILE} \
    [--work-dir ${OUTPUT_DIRECTORY}] \
    [--cfg-options ${OVERRIDE_CONFIGS}]
```

Optional arguments:

- `--work-dir`: If specified, results will be saved in this directory. If not specified, the results will be automatically saved to `work_dirs/{CONFIG_NAME}`.
- `--show`: Show prediction results at runtime, available when `--show-dir` is not specified.
- `--show-dir`: If specified, the visualized segmentation mask will be saved in the specified directory.
- `--wait-time`: The interval of show (s), which takes effect when `--show` is activated. Default to 2.
- `--cfg-options`:  If specified, the key-value pair in xxx=yyy format will be merged into config file.
  For example: To trade speed with GPU memory, you may pass in `--cfg-options model.backbone.with_cp=True` to enable checkpoint in backbone.
  Below is the optional arguments for multi-gpu test:
- `--launcher`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `--local_rank`: ID for local rank. If not specified, it will be set to 0.
  Examples:
  Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test PSPNet on PASCAL VOC (without saving the test results) and evaluate the mIoU.

   ```shell
   python tools/test.py configs/pspnet/pspnet_r50-d8_512x1024_20k_voc12aug.py \
       checkpoints/pspnet_r50-d8_512x1024_20k_voc12aug_20200605_003338-c57ef100.pth
   ```

   Since `--work-dir` is not specified, the folder `work_dirs/pspnet_r50-d8_512x1024_20k_voc12aug` will be created automatically to save the evaluation results.

2. Test PSPNet with 4 GPUs, and evaluate the standard mIoU and cityscapes metric.

   ```shell
   ./tools/dist_test.sh configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
       checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth 4
   ```

:::{note}
There is some gap (~0.1%) between cityscapes mIoU and our mIoU. The reason is that cityscapes average each class with class size by default.
We use the simple version without average for all datasets.
:::
