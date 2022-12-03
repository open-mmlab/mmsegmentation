# 教程4：使用现有模型进行训练和测试

MMsegmentation 支持在多种设备上训练和测试模型。如下文，具体方式分别为单GPU、分布式以及计算集群的训练和测试。通过本教程，你将知晓如何用 MMsegmentation 提供的脚本进行训练和测试。

## 在单GPU上训练和测试

### 在单GPU上训练

`tools/train.py` 文件提供了在单GPU上部署训练任务的方法。

基础用法如下:

```shell
python tools/train.py  ${配置文件} [可选参数]
```

- `--work-dir ${工作路径}`: 重新指定工作路径
- `--amp`: 使用自动混合精度计算
- `--resume`: 从工作路径中保存的最新检查点文件（checkpoint）恢复训练
- `--cfg-options ${需更覆盖的配置}`: 覆盖已载入的配置中的部分设置，并且 以 xxx=yyy 格式的键值对 将被合并到 config 配置文件中。
  比如： '--cfg-option model.encoder.in_channels=6'， 更多细节请看[指导](./1_config.md#Modify-config-through-script-arguments)。

下面是对于多GPU测试的可选参数:

- `--launcher`: 执行器的启动方式。允许选择的参数值有 `none`, `pytorch`, `slurm`, `mpi`。特别的，如果设置为none，测试将非分布式模式下进行。
- `--local_rank`: 分布式中进程的序号。如果没有指定，默认设置为0。

**注意：** 命令行参数 `--resume` 和在配置文件中的参数 `load_from` 的不同之处：

`--resume` 只决定是否继续使用工作路径中最新的checkpoint检查点，它常常用于恢复被意外打断的训练。

`load_from` 会明确指定被载入的checkpoint检查点文件，且训练迭代器将从0开始。这常常用于模型微调。

如果你希望从指定的checkpoint检查点上恢复训练你可以使用：

```python
python tools/train.py ${配置文件} --resume --cfg-options load_from=${检查点}
```

**在CPU上训练**: 如果机器没有GPU，则在CPU上训练的过程是与单GPU训练一致的。如果机器有GPUs但是不希望使用它们，我们只需要在训练前通过以下方式关闭GPUs训练功能。

```shell
export CUDA_VISIBLE_DEVICES=-1
```

然后运行[上方](#training-on-a-single-gpu)脚本。

### 在单GPU上测试

`tools/test.py` 文件提供了在单GPU上部署测试任务的方法。

基础用法如下:

```shell
python tools/test.py ${配置文件} ${checkpoint模型参数文件} [可选参数]
```

这个工具有几个可选参数，包括：

- `--work-dir`: 如果指定，结果会保存在这个路径下。如果没有指定则会保存在`work_dirs/{配置路径config_path}`.
- `--show`: 当`--show-dir`有明确指定时，可以使用该参数，在程序运行过程中显示预测结果。
- `--show-dir`: 分割处理后图像的存储路基。如果该参数有具体指定，则可视化的分割掩码将被保存到 `work_dir/timestamp/show_dir`.
- `--wait-time`: 多次可视化结果的时间间隔。当 `--show` 为激活状态时发挥作用。默认为2。
- `--cfg-options`:  如果被具体指定，以 xxx=yyy 形式的键值对将被合并入配置文件中。

**在CPU上测试**: 如果机器没有GPU，则在CPU上训练的过程是与单GPU训练一致的。如果机器有GPUs但是不希望使用它们，我们只需要在训练前通过以下方式关闭GPUs训练功能。

```shell
export CUDA_VISIBLE_DEVICES=-1
```

然后运行[上方](#testing-on-a-single-gpu)脚本。

## 多GPU、多机器上训练和测试

### 在多GPU上训练

OpenMMLab2.0 通过 `MMDistributedDataParallel`实现 **分布式** 训练。

`tools/dist_train.sh` 文件提供了在在多GPU上部署训练任务的方法。

基础用法如下:

```shell
sh tools/dist_train.sh ${配置文件} ${GPU数量} [可选参数]
```

可选参数与[上方](#training-on-a-single-gpu)相同并且还增加了可以指定gpu数量的参数。

示例:

```shell
# 训练检查点和日志保存在 WORK_DIR=work_dirs/pspnet_r50-d8_4xb4-80k_ade20k-512x512/
# If work_dir is not set, it will be generated automatically.
sh tools/dist_train.sh configs/pspnet/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py 8 --work-dir work_dirs/pspnet_r50-d8_4xb4-80k_ade20k-512x512
```

**注意**: 在训练过程中，检查点和日志保存在同一个文件夹结构下，配置文件在`work_dirs/`下。 as the config file under `work_dirs/`。不推荐自定义的工作路径，因为评估脚本依赖于源自配置文件名的路径。如果你希望将权重保存在其他地方，请用symlink符号链接，例如：

```shell
ln -s ${你的工作路径} ${MMSEG}/work_dirs
```

### 在多GPU上测试

`tools/dist_test.sh` 文件提供了在在多GPU上部署测试任务的方法。

基础用法如下:

```shell
sh tools/dist_test.sh ${配置文件} ${检查点文件} ${GPU数量} [可选参数]
```

可选参数与[上方](#testing-on-a-single-gpu)相同并且还增加了可以指定gpu数量的参数。

示例:

```shell
./tools/dist_test.sh configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth 4
```

### 在单个机器上执行多进程任务

如果你在单个机器上运行多进程任务，比如：需要4个GPU的2进程任务在有8个GPU的单个机器上，你需要为每个任务进程具体指定不同端口（默认29500），从而避免通讯冲突。否则，会有报错信息——`RuntimeError: Address already in use`。

如果你使用 `dist_train.sh` 来进行训练任务，你可以通过调整环境变量 `PORT` 设置端口。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 sh tools/dist_train.sh ${配置文件} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 sh tools/dist_train.sh ${配置文件} 4
```

### Training with multiple machines

MMSegmentation 依赖 `torch.distributed` 包来分布式训练。
因此， 可以通过 PyTorch 的 [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility) 来进行分布式训练。

如果你启动多机器进行训练只用简单地通过以太网连接，你可以直接运行下方命令：

在第一个机器上:

```shell
NNODES=2 NODE_RANK=0 PORT=${主节点端口} MASTER_ADDR=${主节点地址} sh tools/dist_train.sh ${配置文件} ${GPUS}
```

在第二个机器上:

```shell
NNODES=2 NODE_RANK=1 PORT=${主节点端口} MASTER_ADDR=${主节点地址} sh tools/dist_train.sh ${配置文件} ${GPUS}
```

通常，如果你没有使用像InfiniBand一类的高速网络，这个会过程比较慢。

## 通过 Slurm 管理进程

[Slurm](https://slurm.schedmd.com/) 是一个很好的为计算族群提供进程规划的系统。

### 通过 Slurm 在族群上训练

在一个由Slurm管理的族群上，你可以使用来启动训练进程。它支持单节点和多节点的训练。

基础用法如下：

```shell
[GPUS=${GPUS}] sh tools/slurm_train.sh ${划分} ${进程名} ${配置文件} [可选参数]
```

下方是一个通过名为_dev_的Slurm划分，调用4个GPU来训练PSPNet，并设置工作路径为共享文件体系。

```shell
GPUS=4 sh tools/slurm_train.sh dev pspnet configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py --work-dir work_dir/pspnet
```

你可以检查 [the source code](../../../tools/slurm_train.sh) 来查看全部的参数和环境变量。

### 通过 Slurm 在族群上测试

与训练任务相同， MMSegmentation 提供 `slurm_test.sh` 文件来启动测试进程。

基础用法如下：

```shell
[GPUS=${GPUS}] sh tools/slurm_test.sh ${分区} ${进程名} ${配置文件} ${检查点文件} [可选参数]
```

你可以通过 [源码](../../../tools/slurm_test.sh) 来查看全部的参数和环境变量。

**注意：** 使用 Slurm 时，需要设置端口，可从以下方式中选取一种。

1. 我们更推荐的通过`--cfg-options`设置端口，因为这不会改变原始配置：

   ```shell
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${划分} ${进程名} config1.py ${工作路径} --cfg-options env_cfg.dist_cfg.port=29500
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${工作路径} --cfg-options env_cfg.dist_cfg.port=29501
   ```

2. 通过修改config配置文件，设置不同的通讯端口：
   In `config1.py`:

   ```python
   enf_cfg = dict(dist_cfg=dict(backend='nccl', port=29500))
   ```

   在 `config2.py`中：

   ```python
   enf_cfg = dict(dist_cfg=dict(backend='nccl', port=29501))
   ```

   然后你可以通过 config1.py 和 config2.py 同时启动两个任务：

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/slurm_train.sh ${划分} ${进程名} config1.py ${工作路径}
   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/slurm_train.sh ${划分} ${进程名} config2.py ${工作路径}
   ```

3. 在命令行中通过环境变量 `MASTER_PORT` 设置端口 ：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 MASTER_PORT=29500 sh tools/slurm_train.sh ${划分} ${进程名} config1.py ${工作路径}
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 MASTER_PORT=29501 sh tools/slurm_train.sh ${划分} ${进程名} config2.py ${工作路径}
```
