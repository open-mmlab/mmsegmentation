# Efficient Depth Fusion Transformer for Aerial Image Semantic Segmentation

## Abstract
Taking depth into consideration has been proven to improve the performance of semantic segmentation through providing additional geometry information. Most existing works adopt a two-stream network, extracting features from color images and depth images separately using two branches of the same structure, which suffer from high memory and computation costs. We find that depth features acquired by simple downsampling can also play a complementary part in the semantic segmentation task, sometimes even better than the two-stream scheme with the same two branches. In this paper, a novel and efficient depth fusion transformer network for aerial image segmentation is proposed. The presented network utilizes patch merging to downsample depth input and a depth-aware self-attention (DSA) module is designed to mitigate the gap caused by difference between two branches and two modalities. Concretely, the DSA fuses depth features and color features by computing depth similarity and impact on self-attention map calculated by color feature. Extensive experiments on the ISPRS 2D semantic segmentation dataset validate the efficiency and effectiveness of our method. With nearly half the parameters of traditional two-stream scheme, our method acquires 83.82% mIoU on Vaihingen dataset outperforming other state-of-the-art methods and 87.43% mIoU on Potsdam dataset comparable to the state-of-the-art.
<div align="center">
  <img src="resources/EDFT.png"/>
</div>

Paper can be download [here](https://www.mdpi.com/2072-4292/14/5/1294).

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation

## Data
Two [ISPRS Contest](https://www.isprs.org/education/benchmarks/UrbanSemLab/semantic-labeling.aspx) Datasets have been preprocessed to form RGB-D images and organized as a custom of mmsegmentation. Please download from aistudio: [Vaihingen](https://aistudio.baidu.com/aistudio/datasetdetail/103733), [Potsdam](https://aistudio.baidu.com/aistudio/datasetdetail/145287)

## Results

  | DataSet | Backbone | Crop Size | Lr schd | mIoU | mIoU(ms+flip) | config | download |
  | ------ | -------- | --------- | ------: | ---: | ------------- | ------ | -------- |
  |Vaihingen | Segformer-B0 | 256x256 | 80000 | 80.49  | 81.63 |[config](configs/edft/segformer_mit_fuse-b0_256x256_80k_vai.py) | [model](https://pan.baidu.com/s/1AOkmy-P4dv7Ig3rFsinovw)|
  |Vaihingen | Segformer-B1 | 256x256 | 80000 | 81.28 | 82.13 |[config](configs/edft/segformer_mit_fuse-b1_256x256_80k_vai.py) | [model](https://pan.baidu.com/s/1AOkmy-P4dv7Ig3rFsinovw)|
  |Vaihingen | Segformer-B2 | 256x256 | 80000 | 82.17 | 82.88 |[config](configs/edft/segformer_mit_fuse-b2_256x256_80k_vai.py) | [model](https://pan.baidu.com/s/1AOkmy-P4dv7Ig3rFsinovw)|
  |Vaihingen | Segformer-B3 | 256x256 | 80000 | 82.27 | 83.04 |[config](configs/edft/segformer_mit_fuse-b3_256x256_80k_vai.py) | [model](https://pan.baidu.com/s/1AOkmy-P4dv7Ig3rFsinovw)|
  |Vaihingen | Segformer-B4 | 256x256 | 80000 | 83.02 | 83.82 |[config](configs/edft/segformer_mit_fuse-b4_256x256_80k_vai.py) | [model](https://pan.baidu.com/s/1AOkmy-P4dv7Ig3rFsinovw)|
  |Vaihingen | Segformer-B5 | 256x256 | 80000 | 82.48 | 83.23 |[config](configs/edft/segformer_mit_fuse-b5_256x256_80k_vai.py) | [model](https://pan.baidu.com/s/1AOkmy-P4dv7Ig3rFsinovw)|
  |Potsdam | Segformer-B4 | 512x512 | 80000 | 87.22 | 87.40  |[config](configs/edft/segformer_mit_fuse-b4_512x512_80k_pot.py) | [model](https://pan.baidu.com/s/1AOkmy-P4dv7Ig3rFsinovw)|

  password for BaiduNetdisk: dshs 

```
# Single-gpu testing
python tools\test.py configs\edft\segformer_mit_fuse-b0_256x256_80k_vai.py mit_fuse_b0.pth --eval mIoU
```

## Training

```
# Single-gpu training
python tools\train.py configs\edft\segformer_mit_fuse-b0_256x256_80k_vai.py
```

## Citation

```
@Article{rs14051294,
	AUTHOR = {Yan, Li and Huang, Jianming and Xie, Hong and Wei, Pengcheng and Gao, Zhao},
	TITLE = {Efficient Depth Fusion Transformer for Aerial Image Semantic Segmentation},
	JOURNAL = {Remote Sensing},
	VOLUME = {14},
	YEAR = {2022},
	NUMBER = {5},
	ARTICLE-NUMBER = {1294},
	URL = {https://www.mdpi.com/2072-4292/14/5/1294},
	ISSN = {2072-4292},
	DOI = {10.3390/rs14051294}
}
```
