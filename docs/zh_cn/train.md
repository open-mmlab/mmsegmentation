## 训练一个模型

MMSegmentation 可以执行分布式训练和非分布式训练，分别使用 `MMDistributedDataParallel` 和 `MMDataParallel` 命令。

所有的输出(日志 log 和检查点 checkpoints )将被保存到工作路径文件夹里，它可以通过配置文件里的 `work_dir` 指定。

在一定迭代轮次后，我们默认在验证集上评估模型表现。您可以在训练配置文件中添加间隔参数来改变评估间隔。

```python
evaluation = dict(interval=4000)  # 每4000 iterations 评估一次模型的性能
```

**\*重要提示\***: 在配置文件里的默认学习率是针对4卡 GPU 和2张图/GPU (此时 batchsize = 4x2 = 8)来设置的。
同样，您也可以使用8卡 GPU 和 1张图/GPU 的设置，因为所有的模型均使用 cross-GPU 的 SyncBN 模式。

我们可以在训练速度和 GPU 显存之间做平衡。当模型或者 Batch Size 比较大的时，可以传递`--cfg-options model.backbone.with_cp=True` ，使用 `with_cp` 来节省显存，但是速度会更慢，因为原先使用 `ith_cp` 时，是逐层反向传播(Back Propagation, BP)，不会保存所有的梯度。

### 使用单台机器训练

#### 使用单卡 GPU 训练

```shell
python tools/train.py ${CONFIG_FILE} [可选参数]
```

如果您想在命令里定义工作文件夹路径，您可以添加一个参数`--work-dir ${工作路径}`。

#### 使用 CPU 训练

使用 CPU 训练的流程和使用单 GPU 训练的流程一致，我们仅需要在训练流程开始前禁用 GPU。

```shell
export CUDA_VISIBLE_DEVICES=-1
```

之后运行单 GPU 训练脚本即可。

```{warning}
我们不推荐用户使用 CPU 进行训练，这太过缓慢。我们支持这个功能是为了方便用户在没有 GPU 的机器上进行调试。
```

#### 使用多卡 GPU 训练

```shell
sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [可选参数]
```

可选参数可以为:

- `--no-validate` (**不推荐**): 训练时代码库默认会在每 k 轮迭代后在验证集上进行评估，如果不需评估使用命令 `--no-validate`
- `--work-dir ${工作路径}`: 在配置文件里重写工作路径文件夹
- `--resume-from ${检查点文件}`: 继续使用先前的检查点 (checkpoint) 文件（可以继续训练过程）
- `--load-from ${检查点文件}`: 从一个检查点 (checkpoint) 文件里加载权重（对另一个任务进行精调）
- `--deterministic`: 选择此模式会减慢训练速度，但结果易于复现

`resume-from` 和 `load-from` 的区别:

- `resume-from` 加载出模型权重和优化器状态包括迭代轮数等
- `load-from` 仅加载模型权重，从第0轮开始训练

示例:

```shell
# 模型的权重和日志将会存储在这个路径下： WORK_DIR=work_dirs/pspnet_r50-d8_512x512_80k_ade20k/
# 如果work_dir没有被设定，它将会被自动生成
sh tools/dist_train.sh configs/pspnet/pspnet_r50-d8_512x512_80k_ade20k.py 8 --work_dir work_dirs/pspnet_r50-d8_512x512_80k_ade20k/ --deterministic
```

**注意**: 在训练时，模型的和日志保存在“work_dirs/”下的配置文件的相同文件夹结构中。不建议使用自定义的“work_dirs/”，因为验证脚本可以从配置文件名中推断工作目录。如果你想在其他地方保存模型的权重，请使用符号链接，例如:

```shell
ln -s ${YOUR_WORK_DIRS} ${MMSEG}/work_dirs
```

#### 在单个机器上启动多个任务

如果您在单个机器上启动多个任务，例如在8卡 GPU 的一个机器上有2个4卡 GPU 的训练任务，您需要特别对每个任务指定不同的端口（默认为29500）来避免通讯冲突。否则，将会有报错信息 `RuntimeError: Address already in use`。

如果您使用命令 `dist_train.sh` 来启动一个训练任务，您可以在命令行的用环境变量 `PORT` 设置端口:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 sh tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 sh tools/dist_train.sh ${CONFIG_FILE} 4
```

### 使用多台机器训练

如果您想使用由 ethernet 连接起来的多台机器， 您可以使用以下命令:

在第一台机器上:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

但是，如果您不使用高速网路连接这几台机器的话，训练将会非常慢。

### 使用slurm管理任务

Slurm是一个很好的计算集群作业调度系统。在由Slurm管理的集群中，可以使用slurm_train.sh来进行训练。它同时支持单节点和多节点训练。

在多台机器上训练：

```shell
[GPUS=${GPUS}] sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} --work-dir ${WORK_DIR}
```

这里有一个在dev分区上使用16块GPUs来训练PSPNet的例子:

```shell
GPUS=16 sh tools/slurm_train.sh dev pspr50 configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py work_dirs/pspnet_r50-d8_512x1024_40k_cityscapes/
```

当使用 `slurm_train.sh` 在一个节点上启动多个任务时，需要指定不同的端口号，这里提供了三种设置:

方式1：

在`config1.py`中设置:

```python
dist_params = dict(backend='nccl', port=29500)
```

在`config2.py`中设置:

```python
dist_params = dict(backend='nccl', port=29501)
```

然后就可以使用config1.py和config2.py启动两个作业:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2
```

方式2:

您可以设置不同的通信端口，而不需要修改配置文件，但必须设置“cfg-options”，以覆盖配置文件中的默认端口。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1 --cfg-options dist_params.port=29500
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2 --cfg-options dist_params.port=29501
```

方式3:

您可以使用环境变量’ MASTER_PORT ‘在命令中设置端口:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 MASTER_PORT=29500 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 MASTER_PORT=29501 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2
```
