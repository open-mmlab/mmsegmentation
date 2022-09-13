# Models

We usually define a neural network in a deep learning task as a model, and this model is the core of an algorithm. [MMEngine](https://github.com/open-mmlab/mmengine) abstracts a unified model [BaseModel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/base_model.py#L16) to standardize the interfaces for training, testing and other processes. All models implemented by MMSegmentation inherit from `BaseModel`, but we have re-implemented some of the interfaces for semantic segmentation tasks.

## Basic interfaces

MMSegmentation wraps `BaseModel` and implements the [BaseSegmentor](https://github.com/open-mmlab/mmsegmentation/blob/6cdc2c4a8a4f3abc71ee8acdd52c0a46326abe4c/mmseg/models/segmentors/base.py#L15) class, which mainly provides the interfaces `forward`, `train_step`, `val_step` and `test_step`. The following will introduce these interfaces in detail.

### forward

The `forward` method returns losses or predictions of training, validation, testing, and a simple inference process.

The method should accept three modes: "tensor", "predict" and "loss":

- "tensor": Forward the whole network and return the tensor or tuple of tensor without any post-processing, same as a common `nn.Module`.
- "predict": Forward and return the predictions, which are fully processed to a list of `SegDataSample`.
- "loss": Forward and return a `dict` of losses according to the given inputs and data samples.

Note that this method doesn't handle either backpropagation or optimizer updating, which are done in the method `train_step`.

Parameters:

- inputs (torch.Tensor) - The input tensor with shape (N, C, ...) in general.
- data_sample (list\[[SegDataSample](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py)\]) - The seg data samples. It usually includes information such as `metainfo` and `gt_sem_seg`. Default to None.
- mode (str) - Return what kind of value. Defaults to 'tensor'.

Returns:

- `dict` or `list`:
  - If `mode == loss`, return a `dict` of loss tensor used for backward and logging.
  - If `mode == predict`, return a `list` of inference results.
  - If `mode == tensor`, return a `tensor` or `tuple of tensor` or `dict` of `tensor` for custom use.

### train_step

The `train_step` method calls the forward interface of the `loss` mode to get the loss `dict`. The `BaseModel` class implements the default model training process including preprocessing, model forward propagation, loss calculation, optimization, and back-propagation.

Parameters:

- data (dict or tuple or list) - Data sampled from the dataset.
- optim_wrapper (OptimWrapper) - OptimWrapper instance used to update model parameters.

**Note:** [OptimWrapper](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py#L17) provides a common interface for updating parameters, please refer to optimizer wrapper [documentation](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html) in [MMEngine](https://github.com/open-mmlab/mmengine) for more information.

Returns:

- Dict\[str, torch.Tensor\]: A `dict` of tensor for logging.

### val_step

The `val_step` method calls the forward interface of the `predict` mode and returns the prediction result, which is further passed to the process interface of the evaluator and the `after_val_iter` interface of the Hook.

Parameters:

- data (dict or tuple or list) - Data sampled from the dataset.

Returns:

- list - The predictions of given data.

### test_step

The `BaseModel` implements `test_step` the same as `val_step`.

## Data Preprocessor

The [SegDataPreProcessor](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/models/data_preprocessor.py#L13) implemented by MMSegmentation inherits from the [BaseDataPreprocessor](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/data_preprocessor.py#L18) implemented by [MMEngine](https://github.com/open-mmlab/mmengine) and provides the functions of data preprocessing and copying data to the target device.

The runner carries the model to the specified device during the construction stage, while the data is carried to the specified device by the [SegDataPreProcessor](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/models/data_preprocessor.py#L13) in `train_step`, `val_step`, and `test_step`, and the processed data is further passed to the model.

The parameters of the `SegDataPreProcessor` constructor:

- mean (Sequence\[Number\], optional) - The pixel mean of R, G, B channels. Defaults to None.
- std (Sequence\[Number\], optional) - The pixel standard deviation of R, G, B channels. Defaults to None.
- size (tuple, optional) - Fixed padding size.
- size_divisor (int, optional) - The divisor of padded size.
- pad_val (float, optional) - Padding value. Default: 0.
- seg_pad_val (float, optional) - Padding value of segmentation map. Default: 255.
- bgr_to_rgb (bool) - whether to convert image from BGR to RGB. Defaults to False.
- rgb_to_bgr (bool) - whether to convert image from RGB to RGB. Defaults to False.
- batch_augments (list\[dict\], optional) - Batch-level augmentations. Default to None.

The data will be processed as follows:

- Collate and move data to the target device.
- Pad inputs to the input size with defined `pad_val`, and pad seg map with defined `seg_pad_val`.
- Stack inputs to batch_inputs.
- Convert inputs from bgr to rgb if the shape of input is (3, H, W).
- Normalize image with defined std and mean.
- Do batch augmentations like Mixup and Cutmix during training.

The parameters of the `forward` method:

- data (dict) - data sampled from dataloader.
- training (bool) - Whether to enable training time augmentation.

The returns of the `forward` method:

- Dict: Data in the same format as the model input.
