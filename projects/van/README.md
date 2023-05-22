# Visual Attention Network (VAN) for Segmentation

This repo is a PyTorch implementation of applying **VAN** (**Visual Attention Network**) to semantic segmentation.

The code is an integration from [VAN-Segmentation](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/README.md?plain=1)

More details can be found in [**Visual Attention Network**](https://arxiv.org/abs/2202.09741).

## Citation

```bib
@article{guo2022visual,
  title={Visual Attention Network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}
}
```

## Results

**Notes**: Pre-trained models can be found in [TsingHua Cloud](https://cloud.tsinghua.edu.cn/d/0100f0cea37d41ba8d08/).

Results can be found in [VAN-Segmentation](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/README.md?plain=1)

We provide evaluation results of the converted weights.

| Method  |   Backbone   | mIoU  |                                                                    Download                                                                    |
| :-----: | :----------: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| UPerNet |    VAN-B2    | 49.35 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b2-in1kpre_upernet_3rdparty_512x512-ade20k_20230522-19c58aee.pth)  |
| UPerNet |    VAN-B3    | 49.71 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b3-in1kpre_upernet_3rdparty_512x512-ade20k_20230522-653bd6b7.pth)  |
| UPerNet |    VAN-B4    | 51.56 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b4-in1kpre_upernet_3rdparty_512x512-ade20k_20230522-653bd6b7.pth)  |
| UPerNet | VAN-B4-in22k | 52.61 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b4-in22kpre_upernet_3rdparty_512x512-ade20k_20230522-4a4d744a.pth) |
| UPerNet | VAN-B5-in22k | 53.11 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b5-in22kpre_upernet_3rdparty_512x512-ade20k_20230522-5bb6f2b4.pth) |
| UPerNet | VAN-B6-in22k | 54.25 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b6-in22kpre_upernet_3rdparty_512x512-ade20k_20230522-e226b363.pth) |
|   FPN   |    VAN-B0    | 38.65 |   [model](https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b0-in1kpre_fpn_3rdparty_512x512-ade20k_20230522-75a76298.pth)    |
|   FPN   |    VAN-B1    | 43.22 |   [model](https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b1-in1kpre_fpn_3rdparty_512x512-ade20k_20230522-104499ff.pth)    |
|   FPN   |    VAN-B2    | 46.84 |   [model](https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b2-in1kpre_fpn_3rdparty_512x512-ade20k_20230522-7074e6f8.pth)    |
|   FPN   |    VAN-B3    | 48.32 |   [model](https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b3-in1kpre_fpn_3rdparty_512x512-ade20k_20230522-2c3b7f5e.pth)    |

## Preparation

Install MMSegmentation and download ADE20K according to the guidelines in MMSegmentation.

## Requirement

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 1.** Install MMSegmentation.

Case a: If you develop and run mmseg directly, install it from source:

```shell
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

Case b: If you use mmsegmentation as a dependency or third-party package, install it with pip:

```shell
pip install "mmsegmentation>=1.0.0"
```

## Training

If you use 4 GPUs for training by default. Run:

```bash
bash tools/dist_train.sh projects/van/configs/van/van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py 4
```

## Evaluation

To evaluate the model, an example is:

```bash
bash tools/dist_train.sh projects/van/configs/van/van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py work_dirs/van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512/iter_160000.pth 4 --eval mIoU
```

## FLOPs

To calculate FLOPs for a model, run:

```bash
bash tools/analysis_tools/get_flops.py projects/van/configs/van/van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py --shape 512 512
```

## Acknowledgment

Our implementation is mainly based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.12.0), [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation), [PoolFormer](https://github.com/sail-sg/poolformer), [Enjoy-Hamburger](https://github.com/Gsunshine/Enjoy-Hamburger) and [VAN-Segmentation](https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/README.md?plain=1). Thanks for their authors.

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
