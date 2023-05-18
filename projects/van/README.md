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
