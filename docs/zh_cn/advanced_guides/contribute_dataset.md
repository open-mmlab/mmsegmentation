# 在 mmsegmentation projects 中贡献一个标准格式的数据集

- 在开始您的贡献流程前，请先阅读[《OpenMMLab 贡献代码指南》](https://mmcv.readthedocs.io/zh_CN/latest/community/contributing.html)，以详细的了解 OpenMMLab 代码库的代码贡献流程。
- 该教程以 [Gaofen Image Dataset (GID)](https://www.sciencedirect.com/science/article/pii/S0034425719303414) 高分 2 号卫星所拍摄的遥感图像语义分割数据集作为样例，来演示在 mmsegmentation 中的数据集贡献流程。

## 步骤 1： 配置 mmsegmentation 开发所需必要环境

- 开发所必需的环境安装请参考[中文快速入门指南](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/get_started.md)或[英文 get_started](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md)。

- 如果您已安装了最新版的 pytorch、mmcv、mmengine，那么您可以跳过步骤 1 至[步骤 2](<#[步骤-2](#%E6%AD%A5%E9%AA%A4-2%E4%BB%A3%E7%A0%81%E8%B4%A1%E7%8C%AE%E5%89%8D%E7%9A%84%E5%87%86%E5%A4%87%E5%B7%A5%E4%BD%9C)>)。

- **注：** 在此处无需安装 mmsegmentation，只需安装开发 mmsegmentation 所必需的 pytorch、mmcv、mmengine 等即可。

**新建虚拟环境（如已有合适的开发环境，可跳过）**

- 从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda
- 创建一个 conda 环境，并激活

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**安装 pytorch （如环境下已安装 pytorch，可跳过）**

- 参考 [official instructions](https://pytorch.org/get-started/locally/) 安装 **PyTorch**

**使用 mim 安装 mmcv、mmengine**

- 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv)

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

## 步骤 2：代码贡献前的准备工作

### 2.1 Fork mmsegmentation 仓库

- 通过浏览器打开[mmsegmentation 官方仓库](https://github.com/open-mmlab/mmsegmentation/tree/main)。
- 登录您的 GitHub 账户，以下步骤均需在 GitHub 登录的情况下进行。
- Fork mmsegmentation 仓库
  ![image](https://user-images.githubusercontent.com/50650583/233825567-b8bf273c-38f5-4487-b4c6-75ede1e283ee.png)
- Fork 之后，mmsegmentation 仓库将会出现在您的个人仓库中。

### 2.2 在您的代码编写软件中 git clone mmsegmentation

这里以 VSCODE 为例

- 打开 VSCODE，新建终端窗口并激活您在[步骤 1 ](#%E6%AD%A5%E9%AA%A4-1-%E9%85%8D%E7%BD%AE-mmsegmentation-%E5%BC%80%E5%8F%91%E6%89%80%E9%9C%80%E5%BF%85%E8%A6%81%E7%8E%AF%E5%A2%83)中所安装的虚拟环境。
- 在您 GitHub 的个人仓库中找到您 Fork 的 mmsegmentation 仓库，复制其链接。
  ![image](https://github.com/AI-Tianlong/OpenMMLabCamp/assets/50650583/92ad555b-c5b2-4a7f-a800-ebee1e405ab6)
- 在终端中执行命令
  ```bash
  git clone {您所复制的个人仓库的链接}
  ```
  ![image](https://github.com/AI-Tianlong/OpenMMLabCamp/assets/50650583/23ba2636-e66f-4ea5-9077-9dd6b69deb1d)
  **注：** 如提示以下信息，请在 GitHub 中添加 [SSH 秘钥](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
  ![image](https://github.com/AI-Tianlong/OpenMMLabCamp/assets/50650583/6fcab213-0739-483c-b345-c59656027377)
- 进入 mmsegmentation 目录（之后的操作均在 mmsegmentation 目录下）。
  ```bash
  cd mmsegmentation
  ```
- 在终端中执行以下命令，添加官方仓库为上游仓库。
  ```bash
  git remote add upstream git@github.com:open-mmlab/mmsegmentation.git
  ```
- 使用以下命令检查 remote 是否添加成功。
  ```bash
  git remote -v
  ```
  ![image](https://github.com/AI-Tianlong/OpenMMLabCamp/assets/50650583/beec7e5e-2b00-4e49-ab38-f0c79e346594)

### 2.3 切换目录至 mmsegmentation 并从源码安装mmsegmentation

在`mmsegmentation`目录下执行`pip install -v -e .`，通过源码构建方式安装 mmsegmentaion 库。
安装完成后，您将能看到如下图所示的文件树。
<img src="https://user-images.githubusercontent.com/50650583/233826064-4b111358-8f97-44dd-955c-df3204410b8b.png" alt="image" style="zoom:67%;" />

### 2.4 切换分支为 dev-1.x

正如您在[ mmsegmentation 官网](https://github.com/open-mmlab/mmsegmentation/tree/main)所见，该仓库有许多分支，默认分支`main`为稳定的发行版本，以及用于贡献者进行开发的`dev-1.x`分支。`dev-1.x`分支是贡献者们用来提交创意和 PR 的分支，`dev-1.x`分支的内容会被周期性的合入到`main`分支。
![image](https://user-images.githubusercontent.com/50650583/233826225-f4b7299d-de23-47db-900d-dfb01ba0efc3.png)

回到 VSCODE 中，在终端执行命令

```bash
git checkout dev-1.x
```

### 2.5 创新属于自己的新分支

在基于`dev-1.x`分支下，使用如下命令，创建属于您自己的分支。

```bash
# git checkout -b 您的GitHubID/您的分支想要实现的功能的名字
# git checkout -b AI-Tianlong/support_GID_dataset
git checkout -b {您的GitHubID/您的分支想要实现的功能的名字}
```

### 2.6 配置 pre-commit

OpenMMLab 仓库对代码质量有着较高的要求，所有提交的 PR 必须要通过代码格式检查。pre-commit 详细配置参阅[配置 pre-commit](https://mmcv.readthedocs.io/zh_CN/latest/community/contributing.html#pre-commit)。

## 步骤 3：在`mmsegmentation/projects`下贡献您的代码

**先对 GID 数据集进行分析**

这里以贡献高分 2 号遥感图像语义分割数据集 GID 为例，GID 数据集是由我国自主研发的高分 2 号卫星所拍摄的光学遥感图像所创建，经图像预处理后共提供了 150 张 6800x7200 像素的 RGB 三通道遥感图像。并提供了两种不同类别数的数据标注，一种是包含 5 类有效物体的 RGB 标签，另一种是包含 15 类有效物体的 RGB 标签。本教程将针对 5 类标签进行数据集贡献流程讲解。

GID 的 5 类有效标签分别为：0-背景-\[0,0,0\](mask 标签值-标签名称-RGB 标签值)、1-建筑-\[255,0,0\]、2-农田-\[0,255,0\]、3-森林-\[0,0,255\]、4-草地-\[255,255,0\]、5-水-\[0,0,255\]。在语义分割任务中，标签是与原图尺寸一致的单通道图像，标签图像中的像素值为真实样本图像中对应像素所包含的物体的类别。GID 数据集提供的是具有 RGB 三通道的彩色标签，为了模型的训练需要将 RGB 标签转换为 mask 标签。并且由于图像尺寸为 6800x7200 像素，对于神经网络的训练来有些过大，所以将每张图像裁切成了没有重叠的 512x512 的图像以便进行训练。
<img align='center' src="https://user-images.githubusercontent.com/50650583/234192183-83ee4209-e181-4a18-90ca-4d71757cd2c7.png" alt="image" style="zoom:67%;" />

### 3.1 在`mmsegmentation/projects`下创建新的项目文件夹

在`mmsegmentation/projects`下创建文件夹`gid_dataset`
![image](https://user-images.githubusercontent.com/50650583/233829687-8f2b6600-bc9d-48ff-a865-d462af54d55a.png)

### 3.2 贡献您的数据集代码

为了最终能将您在 projects 中贡献的代码更加顺畅的移入核心库中（对代码要求质量更高），非常建议按照核心库的目录来编辑您的数据集文件。
关于数据集有 4 个必要的文件：

- **1**  `mmseg/datasets/gid.py` 定义了数据集的尾缀、CLASSES、PALETTE、reduce_zero_label等
- **2** `configs/_base_/gid.py` GID 数据集的配置文件，定义了数据集的`dataset_type`（数据集类型，`mmseg/datasets/gid.py`中注册的数据集的类名）、`data_root`(数据集所在的根目录，建议将数据集通过软连接的方式将数据集放至`mmsegmentation/data`)、`train_pipline`(训练的数据流)、`test_pipline`(测试和验证时的数据流)、`img_rations`(多尺度预测时的多尺度配置)、`tta_pipeline`（多尺度预测）、`train_dataloader`(训练集的数据加载器)、`val_dataloader`(验证集的数据加载器)、`test_dataloader`(测试集的数据加载器)、`val_evaluator`(验证集的评估器)、`test_evaluator`(测试集的评估器)。
- **3** 使用了 GID 数据集的模型训练配置文件
  这个是可选的，但是强烈建议您添加。在核心库中，所贡献的数据集需要和参考文献中所提出的结果精度对齐，为了后期将您贡献的代码合并入核心库。如您的算力充足，最好能提供对应的模型配置文件在您贡献的数据集上所验证的结果以及相应的权重文件，并撰写较为详细的README.md文档。[示例参考结果](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/deeplabv3plus#mapillary-vistas-v12)
  ![image](https://user-images.githubusercontent.com/50650583/233877682-eabe8723-bce9-40e4-a303-08c8385cb6b5.png)
- **4** 使用如下命令格式： 撰写`docs/zh_cn/user_guides/2_dataset_prepare.md`来添加您的数据集介绍，包括但不限于数据集的下载方式，数据集目录结构、数据集生成等一些必要性的文字性描述和运行命令。以更好地帮助用户能更快的实现数据集的准备工作。

### 3.3 贡献`tools/dataset_converters/gid.py`

由于 GID 数据集是由未经过切分的 6800x7200 图像所构成的数据集，并且没有划分训练集、验证集与测试集。以及其标签为 RGB 彩色标签，需要将标签转换为单通道的 mask label。为了方便训练，首先将 GID 数据集进行裁切和标签转换，并进行数据集划分，构建为 mmsegmentation 所支持的格式。

```python
# tools/dataset_converters/gid.py
import argparse
import glob
import math
import os
import os.path as osp
from PIL import Image

import mmcv
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GID dataset to mmsegmentation format')
    parser.add_argument('dataset_img_path', help='GID images folder path')
    parser.add_argument('dataset_label_path', help='GID labels folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path', default='data/gid')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=256)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args

GID_COLORMAP = dict(
    Background=(0, 0, 0), #0-背景-黑色
    Building=(255, 0, 0), #1-建筑-红色
    Farmland=(0, 255, 0), #2-农田-绿色
    Forest=(0, 0, 255), #3-森林-蓝色
    Meadow=(255, 255, 0),#4-草地-黄色
    Water=(0, 0, 255)#5-水-蓝色
)
palette = list(GID_COLORMAP.values())
classes = list(GID_COLORMAP.keys())

#############用列表来存一个 RGB 和一个类别的对应################
def colormap2label(palette):
    colormap2label_list = np.zeros(256**3, dtype = np.longlong)
    for i, colormap in enumerate(palette):
        colormap2label_list[(colormap[0] * 256 + colormap[1])*256+colormap[2]] = i
    return colormap2label_list

#############给定那个列表，和vis_png然后生成masks_png################
def label_indices(RGB_label, colormap2label_list):
    RGB_label = RGB_label.astype('int32')
    idx = (RGB_label[:, :, 0] * 256 + RGB_label[:, :, 1]) * 256 + RGB_label[:, :, 2]
    # print(idx.shape)
    return colormap2label_list[idx]

def RGB2mask(RGB_label, colormap2label_list):
    # RGB_label = np.array(Image.open(RGB_label).convert('RGB')) #打开RGB_png
    mask_label = label_indices(RGB_label, colormap2label_list) # .numpy()
    return mask_label

colormap2label_list = colormap2label(palette)

def clip_big_image(image_path, clip_save_dir, args, to_label=False):
    """
    Original image of GID dataset is very large, thus pre-processing
    of them is adopted. Given fixed clip size and stride size to generate
    clipped image, the intersection　of width and height is determined.
    For example, given one 6800 x 7200 original image, the clip size is
    256 and stride size is 256, thus it would generate 29 x 27 = 783 images
    whose size are all 256 x 256.

    """

    image = mmcv.imread(image_path, channel_order='rgb')
    # image = mmcv.bgr2gray(image)

    h, w, c = image.shape
    clip_size = args.clip_size
    stride_size = args.stride_size

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * clip_size
    ymin = y * clip_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ], axis=1)

    if to_label:
        image = RGB2mask(image, colormap2label_list) #这里得改一下

    for count, box in enumerate(boxes):
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        img_name = osp.basename(image_path).replace('.tif', '')
        img_name = img_name.replace('_label', '')
        if count % 3 == 0:
            mmcv.imwrite(
                clipped_image.astype(np.uint8),
                osp.join(
                    clip_save_dir.replace('train', 'val'),
                    f'{img_name}_{start_x}_{start_y}_{end_x}_{end_y}.png'))
        else:
            mmcv.imwrite(
                clipped_image.astype(np.uint8),
                osp.join(
                    clip_save_dir,
                    f'{img_name}_{start_x}_{start_y}_{end_x}_{end_y}.png'))
        count += 1

def main():
    args = parse_args()

    """
    According to this paper: https://ieeexplore.ieee.org/document/9343296/
    select 15 images contained in GID, , which cover the whole six
    categories, to generate train set and validation set.

    According to Paper: https://ieeexplore.ieee.org/document/9343296/

    """

    if args.out_dir is None:
        out_dir = osp.join('data', 'gid')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    src_path_list = glob.glob(os.path.join(args.dataset_img_path, '*.tif'))
    print(f'Find {len(src_path_list)} pictures')

    prog_bar = ProgressBar(len(src_path_list))

    dst_img_dir = osp.join(out_dir, 'img_dir', 'train')
    dst_label_dir = osp.join(out_dir, 'ann_dir', 'train')

    for i, img_path in enumerate(src_path_list):
        label_path = osp.join(args.dataset_label_path, osp.basename(img_path.replace('.tif', '_label.tif')))

        clip_big_image(img_path, dst_img_dir, args, to_label=False)
        clip_big_image(label_path, dst_label_dir, args, to_label=True)
        prog_bar.update()

    print('Done!')

if __name__ == '__main__':
    main()
```

### 3.4 贡献`mmseg/datasets/gid.py`

可参考[`projects/mapillary_dataset/mmseg/datasets/mapillary.py`](https://github.com/open-mmlab/mmsegmentation/blob/main/projects/mapillary_dataset/mmseg/datasets/mapillary.py)并在此基础上修改相应变量以适配您的数据集。

```python
# mmseg/datasets/gid.py
# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS

# 注册数据集类
@DATASETS.register_module()
class GID_Dataset(BaseSegDataset):
    """Gaofen Image Dataset (GID)

    Dataset paper link:
    https://www.sciencedirect.com/science/article/pii/S0034425719303414
    https://x-ytong.github.io/project/GID.html

    GID  6 classes: background(others), built-up, farmland, forest, meadow, water

    In This example, select 10 images from GID dataset as training set,
    and select 5 images as validation set.
    The selected images are listed as follows:

    GF2_PMS1__L1A0000647767-MSS1
    GF2_PMS1__L1A0001064454-MSS1
    GF2_PMS1__L1A0001348919-MSS1
    GF2_PMS1__L1A0001680851-MSS1
    GF2_PMS1__L1A0001680853-MSS1
    GF2_PMS1__L1A0001680857-MSS1
    GF2_PMS1__L1A0001757429-MSS1
    GF2_PMS2__L1A0000607681-MSS2
    GF2_PMS2__L1A0000635115-MSS2
    GF2_PMS2__L1A0000658637-MSS2
    GF2_PMS2__L1A0001206072-MSS2
    GF2_PMS2__L1A0001471436-MSS2
    GF2_PMS2__L1A0001642620-MSS2
    GF2_PMS2__L1A0001787089-MSS2
    GF2_PMS2__L1A0001838560-MSS2

    The ``img_suffix`` is fixed to '.tif' and ``seg_map_suffix`` is
    fixed to '.tif' for GID.
    """
    METAINFO = dict(
        classes=('Others', 'Built-up', 'Farmland', 'Forest',
                 'Meadow', 'Water'),

        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255],
                 [255, 255, 0], [0, 0, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=None,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
```

### 3.5 贡献使用 GID 的训练 config file

```python
_base_ = [
    '../../../configs/_base_/models/deeplabv3plus_r50-d8.py',
    './_base_/datasets/gid.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_240k.py'
]
custom_imports = dict(
    imports=['projects.gid_dataset.mmseg.datasets.gid'])

crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6))

```

### 3.6 撰写`docs/zh_cn/user_guides/2_dataset_prepare.md`

**Gaofen Image Dataset (GID)**

- GID 数据集可在[此处](https://x-ytong.github.io/project/Five-Billion-Pixels.html)进行下载。
- GID 数据集包含 150 张 6800x7200 的大尺寸图像，标签为 RGB 标签。
- 此处选择 15 张图像生成训练集和验证集，该 15 张图像包含了所有六类信息。所选的图像名称如下：

```None
  GF2_PMS1__L1A0000647767-MSS1
  GF2_PMS1__L1A0001064454-MSS1
  GF2_PMS1__L1A0001348919-MSS1
  GF2_PMS1__L1A0001680851-MSS1
  GF2_PMS1__L1A0001680853-MSS1
  GF2_PMS1__L1A0001680857-MSS1
  GF2_PMS1__L1A0001757429-MSS1
  GF2_PMS2__L1A0000607681-MSS2
  GF2_PMS2__L1A0000635115-MSS2
  GF2_PMS2__L1A0000658637-MSS2
  GF2_PMS2__L1A0001206072-MSS2
  GF2_PMS2__L1A0001471436-MSS2
  GF2_PMS2__L1A0001642620-MSS2
  GF2_PMS2__L1A0001787089-MSS2
  GF2_PMS2__L1A0001838560-MSS2
```

执行以下命令进行裁切及标签的转换，需要修改为您所存储 15 张图像及标签的路径。

```
python projects/gid_dataset/tools/dataset_converters/gid.py [15 张图像的路径] [15 张标签的路径]
```

完成裁切后的 GID 数据结构如下：

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── gid
│   │   ├── ann_dir
|   │   │   │   ├── train
|   │   │   │   ├── val
│   │   ├── img_dir
|   │   │   │   ├── train
|   │   │   │   ├── val

```

### 3.7 贡献的代码及文档通过`pre-commit`检查

使用命令

```bash
git add .
git commit -m "添加描述"
git push
```

### 3.8 在 GitHub 中向 mmsegmentation 提交 PR

具体步骤可见[《OpenMMLab 贡献代码指南》](https://mmcv.readthedocs.io/zh_CN/latest/community/contributing.html)
