# UNet

> [UniFormer: Unifying Convolution and Self-attention for Visual Recognition](https://arxiv.org/abs/2201.09450)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/Sense-X/UniFormer/tree/main/semantic_segmentation">Official Repo</a>

## Abstract

<!-- [ABSTRACT] -->

It is a challenging task to learn discriminative representation from images and videos, due to large local redundancy and complex global dependency in these visual data. Convolution neural networks (CNNs) and vision transformers (ViTs) have been two dominant frameworks in the past few years. Though CNNs can efficiently decrease local redundancy by convolution within a small neighborhood, the limited receptive field makes it hard to capture global dependency. Alternatively, ViTs can effectively capture long-range dependency via self-attention, while blind similarity comparisons among all the tokens lead to high redundancy. To resolve these problems, we propose a novel Unified transFormer (UniFormer), which can seamlessly integrate the merits of convolution and self-attention in a concise transformer format. Different from the typical transformer blocks, the relation aggregators in our UniFormer block are equipped with local and global token affinity respectively in shallow and deep layers, allowing to tackle both redundancy and dependency for efficient and effective representation learning. Code is available at https://github.com/Sense-X/UniFormer.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/81373517/268431976-1ed6b93a-99cc-4e24-b127-4a5633e60ae3.png" width="100%"/>
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/81373517/268432010-6e4edaeb-f0f8-4ca9-8b91-577388bb846a.png" width="100%"/>
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/81373517/268431897-e28fe62a-db24-4b5d-8739-b4c77820c4bf.jpg" width="100%"/>
</div>

## Citation

```latex
@misc{li2022uniformer,
      title={UniFormer: Unifying Convolution and Self-attention for Visual Recognition},
      author={Kunchang Li and Yali Wang and Junhao Zhang and Peng Gao and Guanglu Song and Yu Liu and Hongsheng Li and Yu Qiao},
      year={2022},
      eprint={2201.09450},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
