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
|Segformer | MIT-B0 | 512x512 | 160000 | - | - | 37.41 | - | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B1 | 512x512 | 160000 | - | - | 41.05 | - | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B2 | 512x512 | 160000 | - | - | 45.68 | - | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B3 | 512x512 | 160000 | - | - | - | - | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B4 | 512x512 | 160000 | - | - | - | - | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B5 | 512x512 | 160000 | - | - | - | - | [config]() | [model]() &#124; [log]() |
