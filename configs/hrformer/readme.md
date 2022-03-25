# HRFormer: High-Resolution Transformer for Dense Prediction

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/HRNet/HRFormer">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.18.0/mmseg/models/backbones/hrformer.py">Code Snippet</a>

<details>
<summary align="right"><a href="https://arxiv.org/abs/2110.09408">HRFormer (NeurIPS'2021)</a></summary>

```latext
@article{YuanFHLZCW21,
  title={HRFormer: High-Resolution Transformer for Dense Prediction},
  author={Yuan, Yuhui and Fu, Rao and Huang, Lang and Lin, Weihong and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  booktitle={NeurIPS},
  year={2021}
}
```

</details>

## Results and models

### Cityscapes

| Method | Backbone           | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) |                                                       config                                                                         | download                                                                                                                                                                                                                                                                                                                                   |
| ------ | ------------------ | --------- | ------: | -------- | -------------- | ----: | ------------: | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| OCRNet |     HRFormer-S     | 512x1024  |  80000  |   11.54  |       1        | 80.68 |    81.65      | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrformer/ocrnet_hrformer-s_4x2_512x1024_80k_cityscapes.py) |  [model]() &#124; [log]()     |
| OCRNet |     HRFormer-B     | 512x1024  |  80000  |   23.80  |       1        | 82.07 |    82.69      | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrformer/ocrnet_hrformer-b_4x2_512x1024_80k_cityscapes.py) |  [model]() &#124; [log]()     |
