# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}
```

## Results and models

### ADE20k

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU | mIoU(ms+flip) | config | download |
| ------ | -------- | --------- | ------: | -------: | -------------- | ---: | ------------- | ------ | -------- |
|Segformer | MIT-B0 | 512x512 | 160000 | - | - | 37.41 | 38.34 | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B1 | 512x512 | 160000 | - | - | 40.97 | 42.54 | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B2 | 512x512 | 160000 | - | - | 45.58 | 47.03 | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B3 | 512x512 | 160000 | - | - | 47.82 | 48.81 | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B4 | 512x512 | 160000 | - | - | 48.46 | 49.76 | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B5 | 640x640 | 160000 | - | - | 49.13 | 50.22 | [config]() | [model]() &#124; [log]() |

Evaluation with AlignedResize:

| Method | Backbone | Crop Size | Lr schd | mIoU | mIoU(ms+flip) |
| ------ | -------- | --------- | ------: | ---: | ------------- |
|Segformer | MIT-B0 | 512x512 | 160000 | 38.1  | 38.57 |
|Segformer | MIT-B1 | 512x512 | 160000 | 41.64 | 42.76 |
|Segformer | MIT-B2 | 512x512 | 160000 | 46.53 | 47.49 |
|Segformer | MIT-B3 | 512x512 | 160000 | 48.46 | 49.14 |
|Segformer | MIT-B4 | 512x512 | 160000 | 49.34 | 50.29 |
|Segformer | MIT-B5 | 640x640 | 160000 | 50.08 | 50.72 |
