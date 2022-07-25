## Inference with pretrained models

We provide testing scripts to evaluate a whole dataset (Cityscapes, PASCAL VOC, ADE20k, etc.),
and also some high-level apis for easier integration to other projects.

### Test a dataset

- single GPU
- CPU
- single node multiple GPU
- multiple node

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}

# CPU: If GPU unavailable, directly running single-gpu testing command above
# CPU: If GPU available, disable GPUs and run single-gpu testing script
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}
```

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test PSPNet on PASCAL VOC (without saving the test results) and evaluate the mIoU.

   ```shell
   python tools/test.py configs/pspnet/pspnet_r50-d8_512x1024_20k_voc12aug.py \
       checkpoints/pspnet_r50-d8_512x1024_20k_voc12aug_20200605_003338-c57ef100.pth
   ```

2. Test PSPNet with 4 GPUs, and evaluate the standard mIoU and cityscapes metric.

   ```shell
   ./tools/dist_test.sh configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
       checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth 4
   ```

:::{note}
There is some gap (~0.1%) between cityscapes mIoU and our mIoU. The reason is that cityscapes average each class with class size by default.
We use the simple version without average for all datasets.
:::
