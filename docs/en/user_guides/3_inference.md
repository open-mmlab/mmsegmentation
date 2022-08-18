# Tutorial 3: Inference with existing models

MMSegmentation provides pre-trained models for semantic segmentation in [Model Zoo](../model_zoo.md), and supports multiple standard datasets, including Cityscapes, ADE20K, etc.
This note will show how to use existing models to inference on given images.
As for how to test existing models on standard datasets, please see this [guide](./4_train_test.md#Test-models-on-standard-datasets)

### Inference on given images

MMSegmentation provides high-level Python APIs for inference on images. Here is an example of building the model and inference on given images.
Please download the [pre-trained model](https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes/pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131-2376f12b.pth) to the path specified by `checkpoint_file` first.

```python
from mmseg.apis import init_model, inference_model
from mmsegseg.utils import register_all_modules
# Specify the path to model config and checkpoint file
config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py'
checkpoint_file = 'checkpoints/pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131-2376f12b.pth'
# register all modules in mmseg into the registries
register_all_modules()
# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')
# test image pair, and save the results
img = 'demo/demo.png'
result = inference_model(model, img)
```
