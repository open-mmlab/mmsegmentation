## 使用预训练模型推理

我们提供测试脚本来评估完整数据集（Cityscapes, PASCAL VOC, ADE20k 等） 上的结果，同时为了使其他项目的整合更容易，也提供一些高级 API。

### 测试一个数据集

- 单卡 GPU
- 单节点多卡 GPU
- 多节点

您可以使用以下命令来测试一个数据集。

```shell
# 单卡 GPU 测试
python tools/test.py ${配置文件} ${检查点文件} [--out ${结果文件}] [--eval ${评估指标}] [--show]

# 多卡GPU 测试
./tools/dist_test.sh ${配置文件} ${检查点文件} ${GPU数目} [--out ${结果文件}] [--eval ${评估指标}]
```

可选参数:

- `RESULT_FILE`: pickle 格式的输出结果的文件名，如果不专门指定，结果将不会被专门保存成文件。
- `EVAL_METRICS`: 在结果里将被评估的指标。这主要取决于数据集，  `mIoU`  对于所有数据集都可获得，像 Cityscapes 数据集可以通过 `cityscapes` 命令来专门评估，就像标准的 `mIoU`一样。
- `--show`: 如果被指定，分割结果将会在一张图像里画出来并且在另一个窗口展示。它仅仅是用来调试与可视化，并且仅针对单卡 GPU 测试。请确认 GUI 在您的环境里可用，否则您也许会遇到报错 `cannot connect to X server`
- `--show-dir`: 如果被指定，分割结果将会在一张图像里画出来并且保存在指定文件夹里。它仅仅是用来调试与可视化，并且仅针对单卡GPU测试。使用该参数时，您的环境不需要 GUI。
- `--eval-options`: 评估时的可选参数，当设置 `efficient_test=True` 时，它将会保存中间结果至本地文件里以节约 CPU 内存。请确认您本地硬盘有足够的存储空间（大于20GB）。

例子:

假设您已经下载检查点文件至文件夹 `checkpoints/` 里。

1. 测试 PSPNet 并可视化结果。按下任何键会进行到下一张图。

    ```shell
    python tools/test.py configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
        checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
        --show
    ```

2. 测试 PSPNet 并保存画出的图以便于之后的可视化。

    ```shell
    python tools/test.py configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
        checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
        --show-dir psp_r50_512x1024_40ki_cityscapes_results
    ```

3. 在数据集 PASCAL VOC (不保存测试结果) 上测试 PSPNet 并评估 mIoU。

    ```shell
    python tools/test.py configs/pspnet/pspnet_r50-d8_512x1024_20k_voc12aug.py \
        checkpoints/pspnet_r50-d8_512x1024_20k_voc12aug_20200605_003338-c57ef100.pth \
        --eval mAP
    ```

4. 使用4卡 GPU 测试 PSPNet，并且在标准 mIoU 和 cityscapes 指标里评估模型。

    ```shell
    ./tools/dist_test.sh configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
        checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
        4 --out results.pkl --eval mIoU cityscapes
    ```

   注意：在 cityscapes mIoU 和我们的 mIoU 指标会有一些差异 (~0.1%) 。因为 cityscapes 默认是根据类别样本数的多少进行加权平均，而我们对所有的数据集都是采取直接平均的方法来得到 mIoU。

5. 在 cityscapes 数据集上4卡 GPU 测试 PSPNet， 并生成 png 文件以便提交给官方评估服务器。

   首先，在配置文件里添加内容： `configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py`，

    ```python
    data = dict(
        test=dict(
            img_dir='leftImg8bit/test',
            ann_dir='gtFine/test'))
    ```

   随后，进行测试。

    ```shell
    ./tools/dist_test.sh configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
        checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
        4 --format-only --eval-options "imgfile_prefix=./pspnet_test_results"
    ```

   您会在文件夹 `./pspnet_test_results` 里得到生成的 png 文件。
   您也许可以运行 `zip -r results.zip pspnet_test_results/` 并提交 zip 文件给 [evaluation server](https://www.cityscapes-dataset.com/submit/)。

6. 在 Cityscapes 数据集上使用 CPU 高效内存选项来测试 DeeplabV3+ `mIoU` 指标 (没有保存测试结果)。

    ```shell
    python tools/test.py \
    configs/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes.py \
    deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth \
    --eval-options efficient_test=True \
    --eval mIoU
    ```

    使用 ```pmap``` 可查看 CPU 内存情况,  ```efficient_test=True``` 会使用约 2.25GB 的 CPU 内存， ```efficient_test=False``` 会使用约 11.06GB 的 CPU 内存。 这个可选参数可以节约很多 CPU 内存。
