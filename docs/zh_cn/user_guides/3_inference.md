# 教程3：使用预训练模型推理

MMSegmentation 在 [Model Zoo](../Model_Zoo.md) 中为语义分割提供了预训练的模型，并支持多个标准数据集，包括 Cityscapes、ADE20K 等。
本说明将展示如何使用现有模型对给定图像进行推理。
关于如何在标准数据集上测试现有模型，请参阅本[指南](./4_train_test.md)

MMSegmentation 为用户提供了数个接口，以便轻松使用预训练的模型进行推理。

- [教程3：使用预训练模型推理](#教程3使用预训练模型推理)
  - [推理器](#推理器)
    - [基本使用](#基本使用)
    - [初始化](#初始化)
    - [可视化预测结果](#可视化预测结果)
    - [模型列表](#模型列表)
  - [推理 API](#推理-api)
    - [mmseg.apis.init_model](#mmsegapisinit_model)
    - [mmseg.apis.inference_model](#mmsegapisinference_model)
    - [mmseg.apis.show_result_pyplot](#mmsegapisshow_result_pyplot)

## 推理器

在 MMSegmentation 中，我们提供了最**方便的**方式 `MMSegInferencer` 来使用模型。您只需 3 行代码就可以获得图像的分割掩膜。

### 基本使用

以下示例展示了如何使用 `MMSegInferencer` 对单个图像执行推理。

```
>>> from mmseg.apis import MMSegInferencer
>>> # 将模型加载到内存中
>>> inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')
>>> # 推理
>>> inferencer('demo/demo.png', show=True)
```

可视化结果应如下所示：

<div align="center">
    <img src='https://user-images.githubusercontent.com/76149310/221507927-ae01e3a7-016f-4425-b966-7b19cbbe494e.png' />
</div>

此外，您可以使用 `MMSegInferencer` 来处理一个包含多张图片的 `list`：

```
# 输入一个图片 list
>>> images = [image1, image2, ...] # image1 可以是文件路径或 np.ndarray
>>> inferencer(images, show=True, wait_time=0.5) # wait_time 是延迟时间，0 表示无限

# 或输入图像目录
>>> images = $IMAGESDIR
>>> inferencer(images, show=True, wait_time=0.5)

# 保存可视化渲染彩色分割图和预测结果
# out_dir 是保存输出结果的目录，img_out_dir 和 pred_out_dir 为 out_dir 的子目录
# 以保存可视化渲染彩色分割图和预测结果
>>> inferencer(images, out_dir='outputs', img_out_dir='vis', pred_out_dir='pred')
```

推理器有一个可选参数 `return_datasamples`，其默认值为 False，推理器的返回值默认为 `dict` 类型，包括 'visualization' 和 'predictions' 两个 key。
如果 `return_datasamples=True` 推理器将返回 [`SegDataSample`](../advanced_guides/structures.md) 或其列表。

```
result = inferencer('demo/demo.png')
# 结果是一个包含 'visualization' 和 'predictions' 两个 key 的 `dict`
# 'visualization' 包含彩色分割图
print(result['visualization'].shape)
# (512, 683, 3)

# 'predictions' 包含带有标签索引的分割掩膜
print(result['predictions'].shape)
# (512, 683)

result = inferencer('demo/demo.png', return_datasamples=True)
print(type(result))
# <class 'mmseg.structures.seg_data_sample.SegDataSample'>

# 输入一个图片 list
results = inferencer(images)
# 输出为列表
print(type(results['visualization']), results['visualization'][0].shape)
# <class 'list'> (512, 683, 3)
print(type(results['predictions']), results['predictions'][0].shape)
# <class 'list'> (512, 683)

results = inferencer(images, return_datasamples=True)
# <class 'list'>
print(type(results[0]))
# <class 'mmseg.structures.seg_data_sample.SegDataSample'>
```

### 初始化

`MMSegInferencer` 必须使用 `model` 初始化，该 `model` 可以是模型名称或一个 `Config`，甚至可以是配置文件的路径。
模型名称可以在模型的元文件（configs/xxx/metafile.yaml）中找到，比如 maskformer 的一个模型名称是 `maskformer_r50-d32_8xb2-160k_ade20k-512x512`，如果输入模型名称，模型的权重将自动下载。以下是其他输入参数：

- weights（str，可选）- 权重的路径。如果未指定，并且模型是元文件中的模型名称，则权重将从元文件加载。默认为 None。
- classes（list，可选）- 输入类别用于结果渲染，由于分割模型的预测结构是标签索引的分割图，`classes` 是一个相应的标签索引的类别列表。若 classes 没有定义，可视化工具将默认使用 `cityscapes` 的类别。默认为 None。
- palette（list，可选）- 输入调色盘用于结果渲染，它是对应分类的配色列表。若 palette 没有定义，可视化工具将默认使用 `cityscapes` 的调色盘。默认为 None。
- dataset_name（str，可选）- [数据集名称或别名](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py#L302-L317)，可视化工具将使用数据集的元信息，如类别和配色，但 `classes` 和 `palette` 具有更高的优先级。默认为 None。
- device（str，可选）- 运行推理的设备。如果无，则会自动使用可用的设备。默认为 None。
- scope（str，可选）- 模型的作用域。默认为 'mmseg'。

### 可视化预测结果

`MMSegInferencer` 有4个用于可视化预测的参数，您可以在初始化推理器时使用它们：

- show（bool）- 是否弹出窗口显示图像。默认为 False。
- wait_time（float）- 显示的间隔。默认值为 0。
- img_out_dir（str）- `out_dir` 的子目录，用于保存渲染有色分割掩膜，因此如果要保存预测掩膜，则必须定义 `out_dir`。默认为 `vis`。
- opacity（int，float）- 分割掩膜的透明度。默认值为 0.8。

这些参数的示例请参考[基本使用](#基本使用)

### 模型列表

在 MMSegmentation 中有一个非常容易列出所有模型名称的方法

```
>>> from mmseg.apis import MMSegInferencer
# models 是一个模型名称列表，它们将自动打印
>>> models = MMSegInferencer.list_models('mmseg')
```

## 推理 API

### mmseg.apis.init_model

从配置文件初始化一个分割器。

参数：

- config（str，`Path` 或 `mmengine.Config`）- 配置文件路径或配置对象。
- checkpoint（str，可选）- 权重路径。如果为 None，则模型将不会加载任何权重。
- device（str，可选）- CPU/CUDA 设备选项。默认为 'cuda:0'。
- cfg_options（dict，可选）- 用于覆盖所用配置中的某些设置的选项。

返回值：

- nn.Module：构建好的分割器。

示例：

```python
from mmseg.apis import init_model

config_path = 'configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_path = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# 初始化不带权重的模型
model = init_model(config_path)

# 初始化模型并加载权重
model = init_model(config_path, checkpoint_path)

# 在 CPU 上的初始化模型并加载权重
model = init_model(config_path, checkpoint_path, 'cpu')
```

### mmseg.apis.inference_model

使用分割器推理图像。

参数：

- model（nn.Module）- 加载的分割器
- imgs（str，np.ndarray 或 list\[str/np.ndarray\]）- 图像文件或加载的图像

返回值：

- `SegDataSample` 或 list\[`SegDataSample`\]：如果 imgs 是列表或元组，则返回相同长度的列表类型结果，否则直接返回分割结果。

**注意：** [SegDataSample](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py) 是 MMSegmentation 的数据结构接口，用作不同组件之间的接口。`SegDataSample` 实现抽象数据元素 `mmengine.structures.BaseDataElement`，请参阅 [MMEngine](https://github.com/open-mmlab/mmengine) 中的数据元素[文档](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/data_element.html)了解更多信息。

`SegDataSample` 中的参数分为几个部分：

- `gt_sem_seg`（`PixelData`）- 语义分割的标注。
- `pred_sem_seg`（`PixelData`）- 语义分割的预测。
- `seg_logits`（`PixelData`）- 模型最后一层的输出结果。

**注意：** [PixelData](https://github.com/open-mmlab/mmengine/blob/main/mmengine/structures/pixel_data.py) 是像素级标注或预测的数据结构，请参阅 [MMEngine](https://github.com/open-mmlab/mmengine) 中的 PixelData [文档](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html)了解更多信息。

示例：

```python
from mmseg.apis import init_model, inference_model

config_path = 'configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_path = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
img_path = 'demo/demo.png'


model = init_model(config_path, checkpoint_path)
result = inference_model(model, img_path)
```

### mmseg.apis.show_result_pyplot

在图像上可视化分割结果。

参数：

- model（nn.Module）- 加载的分割器。
- img（str 或 np.ndarray）- 图像文件名或加载的图像。
- result（`SegDataSample`）- SegDataSample 预测结果。
- opacity（float）- 绘制分割图的不透明度。默认值为 `0.5`，必须在 `(0，1]` 范围内。
- title（str）- pyplot 图的标题。默认值为 ''。
- draw_gt（bool）- 是否绘制 GT SegDataSample。默认为 `True`。
- draw_pred（draws_pred）- 是否绘制预测 SegDataSample。默认为 `True`。
- wait_time（float）- 显示的间隔，0 是表示“无限”的特殊值。默认为 `0`。
- show（bool）- 是否展示绘制的图像。默认为 `True`。
- save_dir（str，可选）- 为所有存储后端保存的文件路径。如果为 `None`，则后端存储将不会保存任何数据。
- out_file（str，可选）- 输出文件的路径。默认为 `None`。

返回值：

- np.ndarray：通道为 RGB 的绘制图像。

示例：

```python
from mmseg.apis import init_model, inference_model, show_result_pyplot

config_path = 'configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_path = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
img_path = 'demo/demo.png'


# 从配置文件和权重文件构建模型
model = init_model(config_path, checkpoint_path, device='cuda:0')

# 推理给定图像
result = inference_model(model, img_path)

# 展示分割结果
vis_image = show_result_pyplot(model, img_path, result)

# 保存可视化结果，输出图像将在 `workdirs/result.png` 路径下找到
vis_iamge = show_result_pyplot(model, img_path, result, out_file='work_dirs/result.png')

# 修改展示图像的时间，注意 0 是表示“无限”的特殊值
vis_image = show_result_pyplot(model, img_path, result, wait_time=5)
```

**注意：** 如果当前设备没有图形用户界面，建议将 `show` 设置为 `False`，并指定 `out_file` 或 `save_dir` 来保存结果。如果您想在窗口上显示结果，则不需要特殊设置。
