# Benchmark and Model Zoo

## Common settings

* We use distributed training with 4 GPUs by default.
* All pytorch-style pretrained backbones on ImageNet are train by ourselves, with the same procedure in the [paper](https://arxiv.org/pdf/1812.01187.pdf).
  Our ResNet style backbone are based on ResNetV1c variant, where the 7x7 conv in the input stem is replaced with three 3x3 convs.
* For the consistency across different hardwares, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 4 GPUs with `torch.backends.cudnn.benchmark=False`.
  Note that this value is usually less than what `nvidia-smi` shows.
* We report the inference time as the total time of network forwarding and post-processing, excluding the data loading time.
  Results are obtained with the script `tools/benchmark.py` which computes the average time on 200 images with `torch.backends.cudnn.benchmark=False`.
* There are two inference modes in this framework.

  * `slide` mode: The `test_cfg` will be like `dict(mode='slide', crop_size=(769, 769), stride=(513, 513))`.

    In this mode, multiple patches will be cropped from input image, passed into network individually.
    The crop size and stride between patches are specified by `crop_size` and `stride`.
    The overlapping area will be merged by average

  * `whole` mode: The `test_cfg` will be like `dict(mode='whole')`.

    In this mode, the whole imaged will be passed into network directly.

    By default, we use `slide` inference for 769x769 trained model, `whole` inference for the rest.
* For input size of 8x+1 (e.g. 769), `align_corner=True` is adopted as a traditional practice.
  Otherwise, for input size of 8x (e.g. 512, 1024), `align_corner=False` is adopted.

## Baselines

### FCN

Please refer to [FCN](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fcn) for details.

### PSPNet

Please refer to [PSPNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/pspnet) for details.

### DeepLabV3

Please refer to [DeepLabV3](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3) for details.

### PSANet

Please refer to [PSANet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/psanet) for details.

### DeepLabV3+

Please refer to [DeepLabV3+](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3plus) for details.

### UPerNet

Please refer to [UPerNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/upernet) for details.

### NonLocal Net

Please refer to [NonLocal Net](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/nlnet) for details.

### EncNet

Please refer to [EncNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/encnet) for details.

### CCNet

Please refer to [CCNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ccnet) for details.

### DANet

Please refer to [DANet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/danet) for details.

### APCNet

Please refer to [APCNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/apcnet) for details.

### HRNet

Please refer to [HRNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet) for details.

### GCNet

Please refer to [GCNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/gcnet) for details.

### DMNet

Please refer to [DMNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dmnet) for details.

### ANN

Please refer to [ANN](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ann) for details.

### OCRNet

Please refer to [OCRNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ocrnet) for details.

### Fast-SCNN

Please refer to [Fast-SCNN](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastscnn) for details.

### ResNeSt

Please refer to [ResNeSt](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/resnest) for details.

### Semantic FPN

Please refer to [Semantic FPN](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/semfpn) for details.

### PointRend

Please refer to [PointRend](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/point_rend) for details.

### MobileNetV2

Please refer to [MobileNetV2](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2) for details.

### MobileNetV3

Please refer to [MobileNetV3](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v3) for details.

### EMANet

Please refer to [EMANet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/emanet) for details.

### DNLNet

Please refer to [DNLNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet) for details.

### CGNet

Please refer to [CGNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/cgnet) for details.

### Mixed Precision (FP16) Training

Please refer [Mixed Precision (FP16) Training](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fp16/README.md) for details.

## Speed benchmark

### Hardware

* 8 NVIDIA Tesla V100 (32G) GPUs
* Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

### Software environment

* Python 3.7
* PyTorch 1.5
* CUDA 10.1
* CUDNN 7.6.03
* NCCL 2.4.08

### Training speed

For fair comparison, we benchmark all implementations with ResNet-101V1c.
The input size is fixed to 1024x512 with batch size 2.

The training speed is reported as followed, in terms of second per iter (s/iter). The lower, the better.

| Implementation | PSPNet (s/iter) | DeepLabV3+ (s/iter) |
|----------------|-----------------|---------------------|
| [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)              | **0.83**       | **0.85**   |
| [SegmenTron](https://github.com/LikeLy-Journey/SegmenTron)                  | 0.84           | 0.85       |
| [CASILVision](https://github.com/CSAILVision/semantic-segmentation-pytorch) | 1.15           | N/A          |
| [vedaseg](https://github.com/Media-Smart/vedaseg)                           | 0.95           | 1.25       |

Note: The output stride of DeepLabV3+ is 8.
