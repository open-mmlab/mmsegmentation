# Benchmark and Model Zoo

## Mirror sites

We use AWS as the main site to host our model zoo, and maintain a mirror on aliyun.
You can replace `https://s3.ap-northeast-2.amazonaws.com/open-mmlab` with `https://open-mmlab.oss-cn-beijing.aliyuncs.com` in model urls.

## Common settings

- All models were trained on `coco_2017_train`, and tested on the `coco_2017_val`.
- We use distributed training.
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo, caffe-style pretrained backbones are converted from the newly released model from detectron2.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.
- We report the inference time as the total time of network forwarding and post-processing, excluding the data loading time. Results are obtained with the script [benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/benchmark.py) which computes the average time on 2000 images.


## Baselines

### FCN

Please refer to [FCN](../configs/fcnnet/README.md) for details.

### PSPNet

Please refer to [PSPNet](../configs/pspnet/README.md) for details.

### PSANet

Please refer to [PSANet](../configs/psanet/README.md) for details.

### DeepLabV3

Please refer to [DeepLabV3](../configs/deeplabv3/README.md) for details.

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
