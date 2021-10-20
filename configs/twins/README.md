# Twins: Revisiting the Design of Spatial Attention in Vision Transformers

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/Meituan-AutoML/Twins">Official Repo</a>

<details>
<summary align="right"><a href="https://arxiv.org/pdf/2104.13840.pdf">Twins (arXiv'2021)</a></summary>

```latex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

</details>

## Usage

``pip install timm==0.3.2``
#### Training

To train Twins-SVT-B on ImageNet  using 8 gpus for 300 epochs, run

```python
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model alt_gvt_base --batch-size 128 --data-path path_to_imagenet --dist-eval --drop-path 0.3
```

#### Evaluation
To evaluate the performance of Twins-SVT-L on ImageNet using one GPU, run

```python
python main.py --eval --resume alt_gvt_large.pth  --model alt_gvt_large --data-path path_to_imagenet
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results and models

### ADE20K

| Model | Alias in the paper | mIoU(ss/ms) | FLOPs(G)|#Params (M) | URL | Log |
| --- | --- | --- | --- | --- |--- |---|
| PCPVT-Small| Twins-PCPVT-S | 46.2/47.5 | 234  | 54.6 | [pcpvt_small.pth](https://drive.google.com/file/d/1PkkBULZZUhIkFKq_D9db1DXUIHwIPlvp/view?usp=sharing) | [pcpvt_s.txt](/logs/upernet_pcpvt_s.txt)
| PCPVT-Base | Twins-PCPVT-B | 47.1/48.4 | 250 | 74.3 | [pcpvt_base.pth](https://drive.google.com/file/d/16sCd0slLLz6xt3C2ma3TkS9rpMS9eezT/view?usp=sharing) | [pcpvt_b.txt](/logs/upernet_pcpvt_b.txt)
| PCPVT-Large| Twins-PCPVT-L | 48.6/49.8 | 269  | 91.5 | [pcpvt_large.pth](https://drive.google.com/file/d/1wsU9riWBiN22fyfsJCHDFhLyP2c_n8sk/view?usp=sharing) | [pcpvt_l.txt](/logs/upernet_pcpvt_l.txt)
| ALTGVT-Small | Twins-SVT-S   | 46.2/47.1 | 228  | 54.4   | [alt_gvt_small.pth](https://drive.google.com/file/d/18OhG0sbAJ5okPj0zn-8YTydKG9jS8TUx/view?usp=sharing) |[svt_s.txt](/logs/upernet_svt_s.txt)
| ALTGVT-Base  | Twins-SVT-B   | 47.4/48.9 | 261  | 88.5   | [alt_gvt_base.pth](https://drive.google.com/file/d/1LNtdvACihmKO6XyBPoJDxbrd6AuHVVvq/view?usp=sharing)|[svt_b.txt](/logs/upernet_svt_b.txt)
| ALTGVT-Large | Twins-SVT-L   | 48.8/50.2 | 297 | 133 | [alt_gvt_large.pth](https://drive.google.com/file/d/1xS91hytfzuMZ5Rgb-W-cOJ9G7ptjVwlO/view?usp=sharing)|[svt_l.txt](/logs/upernet_svt_l.txt)
