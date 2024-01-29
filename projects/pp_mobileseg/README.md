# PP-MobileSeg: Exploring Transformer Blocks for Efficient Mobile Segmentation.

## Reference

> [PP-MobileSeg: Explore the Fast and Accurate Semantic Segmentation Model on Mobile Devices. ](https://arxiv.org/abs/2304.05152)

## Introduction

<a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.8">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/tree/main/projects/pp_mobileseg">Code Snippet</a>

## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> Abstract

With the success of transformers in computer vision, several attempts have been made to adapt transformers to mobile devices. However, their performance is not satisfied for some real world applications. Therefore, we propose PP-MobileSeg, a SOTA semantic segmentation model for mobile devices.

It is composed of three newly proposed parts, the strideformer backbone, the Aggregated Attention Module(AAM), and the Valid Interpolate Module(VIM):

- With the four-stage MobileNetV3 block as the feature extractor, we manage to extract rich local features of different receptive fields with little parameter overhead. Also, we further efficiently empower features from the last two stages with the global view using strided sea attention.
- To effectively fuse the features, we use AAM to filter the detail features with ensemble voting and add the semantic feature to it to enhance the semantic information to the most content.
- At last, we use VIM to upsample the downsampled feature to the original resolution and significantly decrease latency in model inference stage. It only interpolates classes present in the final prediction which only takes around 10% in the ADE20K dataset. This is a common scenario for datasets with large classes. Therefore it significantly decreases the latency of the final upsample process which takes the greatest part of the model's overall latency.

Extensive experiments show that PP-MobileSeg achieves a superior params-accuracy-latency tradeoff compared to other SOTA methods.

<div align="center">
<img src="https://user-images.githubusercontent.com/34859558/227450728-1338fcb1-3b8a-4453-a155-da60abcacb88.png"  width = "1000" />
</div>

## <img src="https://user-images.githubusercontent.com/34859558/190044217-8f6befc2-7f20-473d-b356-148e06265205.png" width="25"/> Performance

### ADE20K

| Model             | Backbone          | Training Iters | Batchsize | Train Resolution | mIoU(%) | latency(ms)\* | params(M) | config                                                                                                                    | Links                                                                                                                                                                                                                                |
| ----------------- | ----------------- | -------------- | --------- | ---------------- | ------- | ------------- | --------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PP-MobileSeg-Base | StrideFormer-Base | 80000          | 32        | 512x512          | 41.57%  | 265.5         | 5.62      | [config](https://github.com/Yang-Changhui/mmsegmentation/tree/add_ppmobileseg/projects/pp_mobileseg/configs/pp_mobileseg) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/pp_mobileseg/pp_mobileseg_mobilenetv3_2xb16_3rdparty-base_512x512-ade20k-f12b44f3.pth)\|[log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_base/train.log) |
| PP-MobileSeg-Tiny | StrideFormer-Tiny | 80000          | 32        | 512x512          | 36.39%  | 215.3         | 1.61      | [config](https://github.com/Yang-Changhui/mmsegmentation/tree/add_ppmobileseg/projects/pp_mobileseg/configs/pp_mobileseg) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/pp_mobileseg/pp_mobileseg_mobilenetv3_2xb16_3rdparty-tiny_512x512-ade20k-a351ebf5.pth)\|[log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_tiny/train.log) |

## Usage

Same as other models in MMsegmentation, you can run the following command to test the model at ${MMSEG_ROOT}:

```shell
./tools/dist_test.sh projects/pp_mobileseg/configs/pp_mobileseg/pp_mobileseg_mobilenetv3_2x16_80k_ade20k_512x512_base.py checkpoints/pp_mobileseg_mobilenetv3_2xb16_3rdparty-base_512x512-ade20k-f12b44f3.pth 8
```

## Inference with ONNXRuntime

### Prerequisites

**1. Install onnxruntime inference engine.**

Choose one of the following ways to install onnxruntime.

- CPU version

```shell
pip install onnxruntime==1.15.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz
tar -zxvf onnxruntime-linux-x64-1.15.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.15.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

**2. Convert model to onnx file**

- Install `mim` and `mmdeploy`.

```shell
pip install openmim
mim install mmdeploy
git clone https://github.com/open-mmlab/mmdeploy.git
```

- Download pp_mobileseg model.

```shell
wget https://download.openmmlab.com/mmsegmentation/v0.5/pp_mobileseg/pp_mobileseg_mobilenetv3_2xb16_3rdparty-tiny_512x512-ade20k-a351ebf5.pth
```

- Convert model to onnx files.

```shell
python mmdeploy/tools/deploy.py mmdeploy/configs/mmseg/segmentation_onnxruntime_dynamic.py \
    configs/pp_mobileseg/pp_mobileseg_mobilenetv3_2x16_80k_ade20k_512x512_tiny.py \
    pp_mobileseg_mobilenetv3_2xb16_3rdparty-tiny_512x512-ade20k-a351ebf5.pth \
    ../../demo/demo.png \
    --work-dir mmdeploy_model/mmseg/ort \
    --show
```

**3. Run demo**

```shell
python inference_onnx.py ${ONNX_FILE_PATH} ${IMAGE_PATH} [${MODEL_INPUT_SIZE} ${DEVICE} ${OUTPUT_IMAGE_PATH}]
```

Example:

```shell
python inference_onnx.py mmdeploy_model/mmseg/ort/end2end.onnx ../../demo/demo.png
```

## Citation

If you find our project useful in your research, please consider citing:

```
@misc{liu2021paddleseg,
      title={PaddleSeg: A High-Efficient Development Toolkit for Image Segmentation},
      author={Yi Liu and Lutao Chu and Guowei Chen and Zewu Wu and Zeyu Chen and Baohua Lai and Yuying Hao},
      year={2021},
      eprint={2101.06175},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{paddleseg2019,
    title={PaddleSeg, End-to-end image segmentation kit based on PaddlePaddle},
    author={PaddlePaddle Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleSeg}},
    year={2019}
}
```
