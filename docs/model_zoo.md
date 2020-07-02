# Benchmark and Model Zoo

## Common settings

* We use distributed training with 4 GPUs.
* All pytorch-style pretrained backbones on ImageNet are train by ourselves, with the same procedure in the [paper](https://arxiv.org/pdf/1812.01187.pdf).
  Our ResNet style backbone are based on ResNetV1c variant, where the 7x7 conv in the input stem is replaced with three 3x3 convs
* For the consistency across different hardwares, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 4 GPUs with `torch.backends.cudnn.benchmark=False`.
  Note that this value is usually less than what `nvidia-smi` shows.
* We report the inference time as the total time of network forwarding and post-processing, excluding the data loading time.
Results are obtained with the script `tools/benchmark.py` which computes the average time on 200 images with `torch.backends.cudnn.benchmark=False`.
* There are two inference mode in this framework.
    * `slide` mode: The `test_cfg` will be like `dict(mode='slide', crop_size=(769, 769), stride=(513, 513))`.

        In this mode, multiple patches will be cropped from input image, passed into network individually.
        The crop size and stride between patches are specified by `crop_size` and `stride`.
        The overlapping area will be merged by average
    * `whole` mode: The `test_cfg` will be like `dict(mode='whole')`.

        In this mode, the whole imaged will be passed into network directly.
* For input size of 8x+1 (e.g. 769), `align_corner=True` is adopted as a traditional practice.
Otherwise, for input size of 8x (e.g. 512, 1024), `align_corner=False` is adopted.


## Baselines

### FCN

Please refer to [FCN](../configs/fcnnet/README.md) for details.

### PSPNet

Please refer to [PSPNet](../configs/pspnet/README.md) for details.

### PSANet

Please refer to [PSANet](../configs/psanet/README.md) for details.

### DeepLabV3

Please refer to [DeepLabV3](../configs/deeplabv3/README.md) for details.

### DeepLabV3+

Please refer to [DeepLabV3+](../configs/deeplabv3plus/README.md) for details.

### UperNet

Please refer to [UperNet](../configs/upernet/README.md) for details.

### HRNet

Please refer to [HRNet](../configs/hrnet/README.md) for details.

### GCNet

Please refer to [gcnet](../configs/gcnet/README.md) for details.

### NonLocal Net

Please refer to [NonLocal Net](../configs/nlnet/README.md) for details.

### CCNet

Please refer to [CCNet](../configs/ccnet/README.md) for details.

### DANet

Please refer to [CCNet](../configs/danet/README.md) for details.

### ANN

Please refer to [ANN](../configs/ann/README.md) for details.

### OCRNet

Please refer to [OCRNet](../configs/ocrnet/README.md) for details.

## Speed benchmark

| Implementation       | Throughput (img/s) |
|----------------------|--------------------|
| [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) | - |

### Hardware

- 8 NVIDIA Tesla V100 (32G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

### Software environment

- Python 3.7
- PyTorch 1.5
- CUDA 10.1
- CUDNN 7.6.03
- NCCL 2.4.08
