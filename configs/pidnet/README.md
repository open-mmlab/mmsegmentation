# PIDNet

> [PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller](https://arxiv.org/pdf/2206.02066.pdf)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/XuJiacong/PIDNet">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/backbones/pidnet.py">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Two-branch network architecture has shown its efficiency and effectiveness for real-time semantic segmentation tasks. However, direct fusion of low-level details and high-level semantics will lead to a phenomenon that the detailed features are easily overwhelmed by surrounding contextual information, namely overshoot in this paper, which limits the improvement of the accuracy of existed two-branch models. In this paper, we bridge a connection between Convolutional Neural Network (CNN) and Proportional-IntegralDerivative (PID) controller and reveal that the two-branch network is nothing but a Proportional-Integral (PI) controller, which inherently suffers from the similar overshoot issue. To alleviate this issue, we propose a novel threebranch network architecture: PIDNet, which possesses three branches to parse the detailed, context and boundary information (derivative of semantics), respectively, and employs boundary attention to guide the fusion of detailed and context branches in final stage. The family of PIDNets achieve the best trade-off between inference speed and accuracy and their test accuracy surpasses all the existed models with similar inference speed on Cityscapes, CamVid and COCO-Stuff datasets. Especially, PIDNet-S achieves 78.6% mIOU with inference speed of 93.2 FPS on Cityscapes test set and 80.1% mIOU with speed of 153.7 FPS on CamVid test set.

<!-- [IMAGE] -->

<div align=center>
<img src="https://raw.githubusercontent.com/XuJiacong/PIDNet/main/figs/pidnet.jpg" width="800"/>
</div>

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device | mIoU  | mIoU(ms+flip) | config                                                                                                                     | download                                                                                                                                                                                                                                                                                                                                                 |
| ------ | -------- | --------- | ------- | -------- | -------------- | ------ | ----- | ------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PIDNet | PIDNet-S | 1024x1024 | 120000  | 3.38     | 80.82          | A100   | 78.74 | 80.87         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes/pidnet-s_2xb6-120k_1024x1024-cityscapes_20230302_191700-bb8e3bcc.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes/pidnet-s_2xb6-120k_1024x1024-cityscapes_20230302_191700.json) |
| PIDNet | PIDNet-M | 1024x1024 | 120000  | 5.14     | 71.98          | A100   | 80.22 | 82.05         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes/pidnet-m_2xb6-120k_1024x1024-cityscapes_20230301_143452-f9bcdbf3.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes/pidnet-m_2xb6-120k_1024x1024-cityscapes_20230301_143452.json) |
| PIDNet | PIDNet-L | 1024x1024 | 120000  | 5.83     | 60.06          | A100   | 80.89 | 82.37         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes/pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes/pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514.json) |

## Notes

The pretrained weights in config files are converted from [the official repo](https://github.com/XuJiacong/PIDNet#models).

## Citation

```bibtex
@misc{xu2022pidnet,
      title={PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller},
      author={Jiacong Xu and Zixiang Xiong and Shankar P. Bhattacharyya},
      year={2022},
      eprint={2206.02066},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
