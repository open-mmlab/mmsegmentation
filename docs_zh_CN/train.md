## 训练一个模型

MMSegmentation 可以执行分布式训练和非分布式训练，分别使用 `MMDistributedDataParallel` 和 `MMDataParallel` 命令。

所有的输出(日志 log 和检查点 checkpoints )将被保存到工作路径文件夹里，它可以通过配置文件里的 `work_dir` 指定。

在一定迭代轮次后，我们默认在验证集上评估模型表现。您可以在训练配置文件中添加间隔参数来改变评估间隔。

```python
evaluation = dict(interval=4000)  # 每4000 iterations 评估一次模型的性能
```

**\*Important\***: 在配置文件里的默认学习率是针对4卡 GPU 和2张图/GPU (此时 batchsize = 4x2 = 8)来设置的。
同样，您也可以使用8卡 GPU 和 1张图/GPU 的设置，因为所有的模型均使用 cross-GPU 的 SyncBN 模式。

我们可以在训练速度和 GPU 显存之间做平衡。当模型或者 Batch Size 比较大的时，可以传递`--options model.backbone.with_cp=True` ，使用 `with_cp` 来节省显存，但是速度会更慢，因为原先使用 `ith_cp` 时，是逐层反向传播(Back Propagation, BP)，不会保存所有的梯度。

### 使用单卡 GPU 训练

```shell
python tools/train.py ${配置文件} [可选参数]
```

如果您想在命令里定义工作文件夹路径，您可以添加一个参数`--work-dir ${YOUR_WORK_DIR}`。

### 使用多卡 GPU 训练

```shell
./tools/dist_train.sh ${配置文件} ${GPU 个数} [可选参数]
```

可选参数可以为:

- `--no-validate` (**不推荐**): 训练时代码库默认会在每 k 轮迭代后在验证集上进行评估，如果不需评估使用命令 `--no-validate`
- `--work-dir ${工作路径}`: 在配置文件里重写工作路径文件夹
- `--resume-from ${检查点文件}`: 继续使用先前的检查点 (checkpoint) 文件（可以继续训练过程）
- `--load-from ${检查点文件}`: 从一个检查点 (checkpoint) 文件里加载权重（对另一个任务进行精调）

`resume-from` 和 `load-from` 的区别:

- `resume-from` 加载出模型权重和优化器状态包括迭代轮数等
- `load-from` 仅加载模型权重，从第0轮开始训练

### 使用多个机器训练

如果您在一个集群上以[slurm](https://slurm.schedmd.com/) 运行 MMSegmentation，
您可以使用脚本 `slurm_train.sh`（这个脚本同样支持单个机器的训练）。

```shell
[GPUS=${GPU 数量}] ./tools/slurm_train.sh ${分区} ${任务名称} ${配置文件} --work-dir ${工作路径}
```

这里是在 dev 分区里使用16块 GPU 训练 PSPNet 的例子。

```shell
GPUS=16 ./tools/slurm_train.sh dev pspr50 configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py /nfs/xxxx/psp_r50_512x1024_40ki_cityscapes
```

您可以查看 [slurm_train.sh](../tools/slurm_train.sh) 以熟悉全部的参数与环境变量。

如果您多个机器已经有以太网连接， 您可以参考 PyTorch
[launch utility](https://pytorch.org/docs/stable/distributed_deprecated.html#launch-utility) 。
若您没有像 InfiniBand 这样高速的网络连接，多机器训练通常会比较慢。

### 在单个机器上启动多个任务

如果您在单个机器上启动多个任务，例如在8卡 GPU 的一个机器上有2个4卡 GPU 的训练任务，您需要特别对每个任务指定不同的端口（默认为29500）来避免通讯冲突。
否则，将会有报错信息 `RuntimeError: Address already in use`。

如果您使用命令 `dist_train.sh` 来启动一个训练任务，您可以在命令行的用环境变量 `PORT` 设置端口。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${配置文件} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${配置文件} 4
```

如果您使用命令 `slurm_train.sh` 来启动训练任务，您可以在命令行的用环境变量 `MASTER_PORT` 设置端口。

```shell
MASTER_PORT=29500 ./tools/slurm_train.sh ${分区} ${任务名称} ${配置文件}
MASTER_PORT=29501 ./tools/slurm_train.sh ${分区} ${任务名称} ${配置文件}
```
