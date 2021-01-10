## Inference with pretrained models

We provide testing scripts to evaluate a whole dataset (Cityscapes, PASCAL VOC, ADE20k, etc.),
and also some high-level apis for easier integration to other projects.

### Test a dataset

- single GPU
- single node multiple GPU
- multiple node

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

Optional arguments:

- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `mIoU` is available for all dataset. Cityscapes could be evaluated by `cityscapes` as well as standard `mIoU` metrics.
- `--show`: If specified, segmentation results will be plotted on the images and shown in a new window. It is only applicable to single GPU testing and used for debugging and visualization. Please make sure that GUI is available in your environment, otherwise you may encounter the error like `cannot connect to X server`.
- `--show-dir`: If specified, segmentation results will be plotted on the images and saved to the specified directory. It is only applicable to single GPU testing and used for debugging and visualization. You do NOT need a GUI available in your environment for using this option.
- `--eval-options`: Optional parameters during evaluation. When `efficient_test=True`, it will save intermediate results to local files to save CPU memory. Make sure that you have enough local storage space (more than 20GB).

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test PSPNet and visualize the results. Press any key for the next image.

    ```shell
    python tools/test.py configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
        checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
        --show
    ```

2. Test PSPNet and save the painted images for latter visualization.

    ```shell
    python tools/test.py configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
        checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
        --show-dir psp_r50_512x1024_40ki_cityscapes_results
    ```

3. Test PSPNet on PASCAL VOC (without saving the test results) and evaluate the mIoU.

    ```shell
    python tools/test.py configs/pspnet/pspnet_r50-d8_512x1024_20k_voc12aug.py \
        checkpoints/pspnet_r50-d8_512x1024_20k_voc12aug_20200605_003338-c57ef100.pth \
        --eval mAP
    ```

4. Test PSPNet with 4 GPUs, and evaluate the standard mIoU and cityscapes metric.

    ```shell
    ./tools/dist_test.sh configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
        checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
        4 --out results.pkl --eval mIoU cityscapes
    ```

   Note: There is some gap (~0.1%) between cityscapes mIoU and our mIoU. The reason is that cityscapes average each class with class size by default.
   We use the simple version without average for all datasets.

5. Test PSPNet on cityscapes test split with 4 GPUs, and generate the png files to be submit to the official evaluation server.

   First, add following to config file `configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py`,

    ```python
    data = dict(
        test=dict(
            img_dir='leftImg8bit/test',
            ann_dir='gtFine/test'))
    ```

   Then run test.

    ```shell
    ./tools/dist_test.sh configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
        checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
        4 --format-only --eval-options "imgfile_prefix=./pspnet_test_results"
    ```

   You will get png files under `./pspnet_test_results` directory.
   You may run `zip -r results.zip pspnet_test_results/` and submit the zip file to [evaluation server](https://www.cityscapes-dataset.com/submit/).

6. CPU memory efficient test DeeplabV3+ on Cityscapes (without saving the test results) and evaluate the mIoU.

    ```shell
    python tools/test.py \
    configs/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes.py \
    deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth \
    --eval-options efficient_test=True \
    --eval mIoU
    ```

    Using ```pmap``` to view CPU memory footprint, it used 2.25GB CPU memory with ```efficient_test=True``` and 11.06GB CPU memory with ```efficient_test=False``` . This optional parameter can save a lot of memory.
