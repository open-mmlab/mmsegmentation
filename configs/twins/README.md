# Twins: Revisiting the Design of Spatial Attention in Vision Transformers

## Introduction

<!-- [ALGORITHM] -->

<a href = "https://github.com/Meituan-AutoML/Twins">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/faketwins.py#L524">Code Snippet</a>

<details>
<summary align = "right"> <a href = "https://arxiv.org/pdf/2104.13840.pdf" >Twins (arXiv'2021)</a></summary>

```latex
@article{liu2021Swin,
         title = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
         author = {Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
         journal = {arXiv preprint arXiv: 2103.14030},
         year = {2021}
         }
```

</details>

To train Twins - SVT - B on ImageNet  using 8 gpus for 300 epochs, run

```python
python - m torch.distributed.launch - -nproc_per_node = 8 - -use_env main.py - -model alt_gvt_base - -batch - size 128 - -data - path path_to_imagenet - -dist - eval - -drop - path 0.3
```

## Evaluation

To evaluate the performance of Twins - SVT - L on ImageNet using one GPU, run

```python
python main.py - -eval - -resume alt_gvt_large.pth - -model alt_gvt_large - -data - path path_to_imagenet
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results and models

### ADE20K

|   Method    |   Backbone    | Crop Size  |  Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config | download |
|   ------    |   --------    | ---------  |  ------   | -------------- | ----- | ------------- | ------ |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PCPVT-Small | Twins-PCPVT-S | 512x512    |  ------   |    -------     | 46.2  | 47.5          |  --    |  --   |
| PCPVT-Base  | Twins-PCPVT-B | 512x512    |  ------   |    -------     | 47.1  | 48.4          |  --    |  --  |
| PCPVT-Large | Twins-PCPVT-L | 512x512    |  ------   |    -------     | 48.6  | 49.8          |  --    |  --  |
| ALTGVT-Small| Twins-SVT-S   | 512x512    |  ------   |    -------     | 46.2  | 47.1          |  --    |  --  |
| ALTGVT-Base | Twins-SVT-B   | 512x512    |  ------   |    -------     | 47.4  | 48.9          |  --    |  --  |
| ALTGVT-Large| Twins-SVT-L   | 512x512    |  ------   |    -------     | 48.8  | 50.2          |  --    |  --  |
