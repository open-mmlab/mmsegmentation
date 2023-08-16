# AdaBins: Depth Estimation Using Adaptive Bins

## Reference

>  [AdaBins: Depth Estimation Using Adaptive Bins](https://arxiv.org/abs/2011.14141)

## Introduction
<a href="https://github.com/shariqfarooq123/AdaBins">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x/projects/Adabins">Code Snippet</a>

## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> Abstract
We address the problem of estimating a high quality dense depth map from a single RGB input image. We start out with a baseline encoder-decoder convolutional neural network architecture and pose the question of how the global processing of information can help improve overall depth estimation. To this end, we propose a transformer-based architecture block that divides the depth range into bins whose center value is estimated adaptively per image. The final depth values are estimated as linear combinations of the bin centers. We call our new building block AdaBins. Our results show a decisive improvement over the state-of-the-art on several popular depth datasets across all metrics.We also validate the effectiveness of the proposed block with an ablation study and provide the code and corresponding pre-trained weights of the new state-of-the-art model.

Our main contributions are the following:
• We propose an architecture building block that performs global processing of the scene’s information.We propose to divide the predicted depth range into bins where the bin widths change per image. The final depth estimation is a linear combination of the bin center values.
• We show a decisive improvement for supervised single image depth estimation across all metrics for the two most popular datasets, NYU [36] and KITTI [13].
• We analyze our findings and investigate different modifications on the proposed AdaBins block and study
their effect on the accuracy of the depth estimation.

![image-20230816181703680](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230816181703680.png)


## <img src="https://user-images.githubusercontent.com/34859558/190044217-8f6befc2-7f20-473d-b356-148e06265205.png" width="25"/> Performance

### NYU and KITTI
| Model | Encoder | Training epoch | Batchsize | Train Resolution | δ1 | δ2 | δ3 | REL | RMS | RMS log | params(M) | Links |
|-|-|-|-|-|-|-|-|-|-|-|-|-|
|AdaBins_nyu|EfficientNet-B5|25|16|416x544|0.903| 0.984 | 0.997 | 0.103 | 0.364 | 0.044   |78|[model]()|
|AdaBins_kitti|EfficientNet-B5|25|16|352x764|0.964| 0.995 | 0.999 | 0.058 | 2.360 | 0.088   |78|[model]()|



