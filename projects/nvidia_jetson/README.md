# 将 MMSeg 模型调优及部署到 NVIDIA Jetson 平台教程
本教程所用 mmsegmentation 版本： v1.1.1  
本教程所用 NVIDIA Jetson 设备：NVIDIA Jetson AGX Orin 64G
<div align="center">
    <img src="https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/b5466cfd-71a9-4e06-9823-c253a97d57b5" alt="Smiley face" width="50%">  
</div>

## 1 配置 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
* 根据[安装和验证](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/get_started.md)文档，完成开发 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 所需的 `[pytorch](https://pytorch.org/get-started/locally/)`、`mmcv`、`mmengine` 等环境依赖安装。
* 从 GitHub 使用 git clone 命令完成 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 下载。网络不好的同学，可通过 [MMSeg GitHub](https://github.com/open-mmlab/mmsegmentation) 页面进行 zip 的下载。
    ```bash
    git clone https://github.com/open-mmlab/mmsegmentation.git
    ```
* 使用 `pip install -v -e.` 命令动态安装 mmsegmentation 。
    ```bash 
    cd mmsegmentation
    pip install -v -e .
    ```
    提示成功安装后，可通过 `pip list` 命令查看到 mmsegmentation 已通过本地安装方式安装到了您的环境中。
    ![mmseg-install](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/a9c7bcc9-cdcc-40a4-bd7b-8153195549c8)

## 2 准备您的数据集
* 本教程使用遥感图像语义分割数据集 [potsdam](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/user_guides/2_dataset_prepare.md#isprs-potsdam) 作为示例。
* 根据[potsdam 数据准备](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/user_guides/2_dataset_prepare.md#isprs-potsdam)文档，进行数据集下载及 mmseg 格式的准备。
* potsdam 数据集是以德国一个典型的历史城市 Potsdam 命名的，该城市有着大建筑群、狭窄的街道和密集的建筑结构。 potsdam 数据集包含 38 幅 6000x6000 像素的图像，该数据的空间分辨率为 5cm，该数据集的示例如下图：
    ![potsdam-img](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/3bc0a75b-1693-4ae6-aeea-ad502e955068)
