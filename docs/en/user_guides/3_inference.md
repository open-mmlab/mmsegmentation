# Tutorial 3: Inference with existing models

MMSegmentation provides pre-trained models for semantic segmentation in [Model Zoo](../model_zoo.md), and supports multiple standard datasets, including Cityscapes, ADE20K, etc.
This note will show how to use existing models to inference on given images.
As for how to test existing models on standard datasets, please see this [guide](./4_train_test.md)

## Inference API

MMSegmentation provides several interfaces for users to easily use pre-trained models for inference.

- [mmseg.apis.init_model](#mmsegapisinit_model)
- [mmseg.apis.inference_model](#mmsegapisinference_model)
- [mmseg.apis.show_result_pyplot](#mmsegapisshow_result_pyplot)

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
from mmseg.utils import register_all_modules

config_path = 'configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_path = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# register all modules in mmseg into the registries
register_all_modules()

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
from mmseg.utils import register_all_modules

config_path = 'configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_path = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
img_path = 'demo/demo.png'

# register all modules in mmseg into the registries
register_all_modules()

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
from mmseg.utils import register_all_modules

config_path = 'configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_path = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
img_path = 'demo/demo.png'

# register all modules in mmseg into the registries
register_all_modules()

# build the model from a config file and a checkpoint file
model = init_model(config_path, checkpoint_path, device='cuda:0')

# inference on given image
result = inference_model(model, img_path)

# display the segmentation result
vis_image = show_result_pyplot(model, img_path, result)

# save the visualization result, the output image would be found at the path `work_dirs/result.png`
vis_iamge = show_result_pyplot(model, img_path, result, out_file='work_dirs/result.png')

# Modify the time of displaying images, note that 0 is the special value that means "forever".
vis_image = show_result_pyplot(model, img_path, result, wait_time=5)
```

**Note:** If your current device doesn't have graphical user interface, it is recommended that setting `show` to `False` and specify the `out_file` or `save_dir` to save the results. If you would like to display the result on a window, no special settings are required.
