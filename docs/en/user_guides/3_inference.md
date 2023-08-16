# Tutorial 3: Inference with existing models

MMSegmentation provides pre-trained models for semantic segmentation in [Model Zoo](../model_zoo.md), and supports multiple standard datasets, including Cityscapes, ADE20K, etc.
This note will show how to use existing models to inference on given images.
As for how to test existing models on standard datasets, please see this [guide](./4_train_test.md)

MMSegmentation provides several interfaces for users to easily use pre-trained models for inference.

- [Tutorial 3: Inference with existing models](#tutorial-3-inference-with-existing-models)
  - [Inferencer](#inferencer)
    - [Basic Usage](#basic-usage)
    - [Initialization](#initialization)
    - [Visualize prediction](#visualize-prediction)
    - [List model](#list-model)
  - [Inference API](#inference-api)
    - [mmseg.apis.init_model](#mmsegapisinit_model)
    - [mmseg.apis.inference_model](#mmsegapisinference_model)
    - [mmseg.apis.show_result_pyplot](#mmsegapisshow_result_pyplot)

## Inferencer

We provide the most **convenient** way to use the model in MMSegmentation `MMSegInferencer`. You can get segmentation mask for an image with only 3 lines of code.

### Basic Usage

The following example shows how to use `MMSegInferencer` to perform inference on a single image.

```
>>> from mmseg.apis import MMSegInferencer
>>> # Load models into memory
>>> inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')
>>> # Inference
>>> inferencer('demo/demo.png', show=True)
```

The visualization result should look like:

<div align="center">
    <img src='https://user-images.githubusercontent.com/76149310/221507927-ae01e3a7-016f-4425-b966-7b19cbbe494e.png' />
</div>

Moreover, you can use `MMSegInferencer` to process a list of images:

```
# Input a list of images
>>> images = [image1, image2, ...] # image1 can be a file path or a np.ndarray
>>> inferencer(images, show=True, wait_time=0.5) # wait_time is delay time, and 0 means forever

# Or input image directory
>>> images = $IMAGESDIR
>>> inferencer(images, show=True, wait_time=0.5)

# Save visualized rendering color maps and predicted results
# out_dir is the directory to save the output results, img_out_dir and pred_out_dir are subdirectories of out_dir
# to save visualized rendering color maps and predicted results
>>> inferencer(images, out_dir='outputs', img_out_dir='vis', pred_out_dir='pred')
```

There is a optional parameter of inferencer, `return_datasamples`, whose default value is False, and return value of inferencer is a `dict` type by default, including 2 keys 'visualization' and 'predictions'.
If `return_datasamples=True` inferencer will return [`SegDataSample`](../advanced_guides/structures.md), or list of it.

```
result = inferencer('demo/demo.png')
# result is a `dict` including 2 keys 'visualization' and 'predictions'
# 'visualization' includes color segmentation map
print(result['visualization'].shape)
# (512, 683, 3)

# 'predictions' includes segmentation mask with label indice
print(result['predictions'].shape)
# (512, 683)

result = inferencer('demo/demo.png', return_datasamples=True)
print(type(result))
# <class 'mmseg.structures.seg_data_sample.SegDataSample'>

# Input a list of images
results = inferencer(images)
# The output is list
print(type(results['visualization']), results['visualization'][0].shape)
# <class 'list'> (512, 683, 3)
print(type(results['predictions']), results['predictions'][0].shape)
# <class 'list'> (512, 683)

results = inferencer(images, return_datasamples=True)
# <class 'list'>
print(type(results[0]))
# <class 'mmseg.structures.seg_data_sample.SegDataSample'>
```

### Initialization

`MMSegInferencer` must be initialized from a `model`, which can be a model name or a `Config` even a path of config file.
The model names can be found in models' metafile (configs/xxx/metafile.yaml), like one model name of maskformer is `maskformer_r50-d32_8xb2-160k_ade20k-512x512`, and if input model name and the weights of the model will be download automatically. Below are other input parameters:

- weights (str, optional) -  Path to the checkpoint. If it is not specified and model is a model name of metafile, the weights will be loaded from metafile. Defaults to None.
- classes (list, optional) - Input classes for result rendering, as the prediction of segmentation model is a segment map with label indices, `classes` is a list which includes items responding to the label indices. If classes is not defined, visualizer will take `cityscapes` classes by default. Defaults to None.
- palette (list, optional) - Input palette for result rendering, which is a list of colors responding to the classes. If the palette is not defined, the visualizer will take the palette of `cityscapes` by default. Defaults to None.
- dataset_name (str, optional) - [Dataset name or alias](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py#L302-L317), visualizer will use the meta information of the dataset i.e. classes and palette, but the `classes` and `palette` have higher priority. Defaults to None.
- device (str, optional) - Device to run inference. If None, the available device will be automatically used. Defaults to None.
- scope (str, optional) - The scope of the model. Defaults to 'mmseg'.

### Visualize prediction

`MMSegInferencer` supports 4 parameters for visualize prediction, you can use them when call initialized inferencer:

- show (bool) - Whether to display the image in a popup window. Defaults to False.
- wait_time (float) - The interval of show (s). Defaults to 0.
- img_out_dir (str) - Subdirectory of `out_dir`, used to save rendering color segmentation mask, so `out_dir` must be defined if you would like to save predicted mask. Defaults to 'vis'.
- opacity (int, float) - The transparency of segmentation mask. Defaults to 0.8.

The examples of these parameters is in [Basic Usage](#basic-usage)

### List model

There is a very easy to list all model names in MMSegmentation

```
>>> from mmseg.apis import MMSegInferencer
# models is a list of model names, and them will print automatically
>>> models = MMSegInferencer.list_models('mmseg')
```

## Inference API

### mmseg.apis.init_model

Initialize a segmentor from config file.

Parameters:

- config (str, `Path`, or `mmengine.Config`) - Config file path or the config object.
- checkpoint (str, optional) - Checkpoint path. If left as None, the model will not load any weights.
- device (str, optional) - CPU/CUDA device option. Default 'cuda:0'.
- cfg_options (dict, optional) - Options to override some settings in the used config.

Returns:

- nn.Module: The constructed segmentor.

Example:

```python
from mmseg.apis import init_model

config_path = 'configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_path = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# initialize model without checkpoint
model = init_model(config_path)

# init model and load checkpoint
model = init_model(config_path, checkpoint_path)

# init model and load checkpoint on CPU
model = init_model(config_path, checkpoint_path, 'cpu')
```

### mmseg.apis.inference_model

Inference image(s) with the segmentor.

Parameters:

- model (nn.Module) - The loaded segmentor
- imgs (str, np.ndarray, or list\[str/np.ndarray\]) - Either image files or loaded images

Returns:

- `SegDataSample` or list\[`SegDataSample`\]: If imgs is a list or tuple, the same length list type results will be returned, otherwise return the segmentation results directly.

**Note:** [SegDataSample](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py) is a data structure interface of MMSegmentation, it is used as interfaces between different components. `SegDataSample` implement the abstract data element `mmengine.structures.BaseDataElement`, please refer to data element [documentation](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html) in [MMEngine](https://github.com/open-mmlab/mmengine) for more information.

The attributes in `SegDataSample` are divided into several parts:

- `gt_sem_seg` (`PixelData`) - Ground truth of semantic segmentation.
- `pred_sem_seg` (`PixelData`) - Prediction of semantic segmentation.
- `seg_logits` (`PixelData`) - Predicted logits of semantic segmentation.

**Note** [PixelData](https://github.com/open-mmlab/mmengine/blob/main/mmengine/structures/pixel_data.py) is the data structure for pixel-level annotations or predictions, please refer to PixelData [documentation](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html) in [MMEngine](https://github.com/open-mmlab/mmengine) for more information.

Example:

```python
from mmseg.apis import init_model, inference_model

config_path = 'configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_path = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
img_path = 'demo/demo.png'


model = init_model(config_path, checkpoint_path)
result = inference_model(model, img_path)
```

### mmseg.apis.show_result_pyplot

Visualize the segmentation results on the image.

Parameters:

- model (nn.Module) - The loaded segmentor.
- img (str or np.ndarray) - Image filename or loaded image.
- result (`SegDataSample`) - The prediction SegDataSample result.
- opacity (float) - Opacity of painted segmentation map. Default `0.5`, must be in `(0, 1]` range.
- title (str) - The title of pyplot figure. Default is ''.
- draw_gt (bool) - Whether to draw GT SegDataSample. Default to `True`.
- draw_pred (draws_pred) - Whether to draw Prediction SegDataSample. Default to `True`.
- wait_time (float) - The interval of show (s), 0 is the special value that means "forever". Default to `0`.
- show (bool) - Whether to display the drawn image. Default to `True`.
- save_dir (str, optional) - Save file dir for all storage backends. If it is `None`, the backend storage will not save any data.
- out_file (str, optional) - Path to output file. Default to `None`.

Returns:

- np.ndarray: the drawn image which channel is RGB.

Example:

```python
from mmseg.apis import init_model, inference_model, show_result_pyplot

config_path = 'configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_path = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
img_path = 'demo/demo.png'


# build the model from a config file and a checkpoint file
model = init_model(config_path, checkpoint_path, device='cuda:0')

# inference on given image
result = inference_model(model, img_path)

# display the segmentation result
vis_image = show_result_pyplot(model, img_path, result)

# save the visualization result, the output image would be found at the path `work_dirs/result.png`
vis_iamge = show_result_pyplot(model, img_path, result, out_file='work_dirs/result.png')

# Modify the time of displaying images, note that 0 is the special value that means "forever"
vis_image = show_result_pyplot(model, img_path, result, wait_time=5)
```

**Note:** If your current device doesn't have graphical user interface, it is recommended that setting `show` to `False` and specify the `out_file` or `save_dir` to save the results. If you would like to display the result on a window, no special settings are required.
