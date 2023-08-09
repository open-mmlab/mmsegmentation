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

| Model             | Backbone          | Training Iters | Batchsize | Train Resolution | mIoU(%) | latency(ms)\* | params(M) | config                                                                                                                    | Links                                                                                                                                                                                                                          |
| ----------------- | ----------------- | -------------- | --------- | ---------------- | ------- | ------------- | --------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PP-MobileSeg-Base | StrideFormer-Base | 80000          | 32        | 512x512          | 41.57%  | 265.5         | 5.62      | [config](https://github.com/Yang-Changhui/mmsegmentation/tree/add_ppmobileseg/projects/pp_mobileseg/configs/pp_mobileseg) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/pp_mobileseg/pp_mobileseg_mobilenetv3_2x16_80k_ade20k_512x512_base-ed0be681.pth)\|[log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_base/train.log) |
| PP-MobileSeg-Tiny | StrideFormer-Tiny | 80000          | 32        | 512x512          | 36.39%  | 215.3         | 1.61      | [config](https://github.com/Yang-Changhui/mmsegmentation/tree/add_ppmobileseg/projects/pp_mobileseg/configs/pp_mobileseg) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/pp_mobileseg/pp_mobileseg_mobilenetv3_2x16_80k_ade20k_512x512_tiny-e4b35e96.pth)\|[log](https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_tiny/train.log) |

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
