# Twins: Revisiting the Design of Spatial Attention in Vision Transformers

## Introduction

<!-- [ALGORITHM] -->

<a href = "https://github.com/Meituan-AutoML/Twins">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/faketwins.py#L524">Code Snippet</a>

<details>
<summary align = "right"> <a href = "https://arxiv.org/pdf/2104.13840.pdf" >Twins (arXiv'2021)</a></summary>

```latex
@article{chu2021twins,
  title={Twins: Revisiting spatial attention design in vision transformers},
  author={Chu, Xiangxiang and Tian, Zhi and Wang, Yuqing and Zhang, Bo and Ren, Haibing and Wei, Xiaolin and Xia, Huaxia and Shen, Chunhua},
  journal={arXiv preprint arXiv:2104.13840},
  year={2021}altgvt
}
```

</details>

## Usage

To use other repositories' pre-trained models, it is necessary to convert keys.

We provide a script [`twins2mmseg.py`](../../tools/model_converters/twins2mmseg.py) in the tools directory to convert the key of models from [the official repo](https://github.com/Meituan-AutoML/Twins) to MMSegmentation style.

```shell
python tools/model_converters/twins2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results and models

### ADE20K

| Method| Backbone | Crop Size  | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config | download |
| ----- | ------- | ---------  |  ------|  ------   | -------------- | ----- | ------------- | ------ |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Twins(8x4) | PCPVT-S | 512x512    |  160000|  ------   |    -------     | 46.2  | 47.5          |  --    |  --   |
| Twins | PCPVT-B | 512x512    |  160000|  ------   |    -------     | 47.1  | 48.4          |  --    |  --  |
| Twins | PCPVT-L | 512x512    |  160000|  ------   |    -------     | 48.6  | 49.8          |  --    |  --  |
| Twins | SVT-S| 512x512    |  160000|  ------   |    -------     | 46.2  | 47.1          |  --    |  --  |
| Twins | SVT-B| 512x512    |  160000|  ------   |    -------     | 47.4  | 48.9          |  --    |  --  |
| Twins | SVT-L| 512x512    |  160000|  ------   |    -------     | 48.8  | 50.2          |  --    |  --  |

Note:

- `8x2` means 8 GPUs with 4 samples per GPU in training.
