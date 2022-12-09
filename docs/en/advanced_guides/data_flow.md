# Dataflow

In this chapter, we will introduce the dataflow and data format convention between the internal modules managed by the [Runner](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html).

## Overview of dataflow

As illustrated in the [Runner document of MMEngine](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html), the following diagram shows the basic dataflow.

![Basic dataflow](https://user-images.githubusercontent.com/112053249/199228350-5f80699e-7fd2-4b4c-ac32-0b16b1922c2e.png)

The dashed border, gray filled shapes represent different data formats, while solid boxes represent modules/methods. Due to the great flexibility and extensibility of MMEngine, you can always inherit some key base classes and override their methods, so the above diagram doesn’t always hold. It only holds when you are not customizing your own `Runner` or `TrainLoop`, and you are not overriding `train_step`, `val_step` or `test_step` method in your custom model.

## Format convention

### DataLoader to DataPreprocessor

DataLoader is a necessary component in MMEngine's training and testing pipelines. It's conceptually derived from and consistent with PyTorch. DataLoader loads data from filesystem and the original data pass through data prepare pipeline then would be sent to the DataPreprocessor. MMSegmentation defines the default data format at [PackSegInputs](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/datasets/transforms/formatting.py#L12), it's the last component of `train_pipeline` and `test_pipeline`.

Without any modifications, the return value of PackSegInputs is usually a `dict` and has only two keys, `inputs` and `data_samples`. The following pseudo-code shows the data types of the values corresponding to the two keys, `inputs` is the input tenor to the model and `data_samples` contains the meta information of input images.

```python
dict(
    inputs=torch.Tensor,
    data_samples=SegDataSample
)
```

**Note:** [SegDataSample](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py) is a data structure interface of MMSegmentation, it is used as an interface between different components. `SegDataSample` implements the abstract data element `mmengine.structures.BaseDataElement`, please refer to [the SegDataSample documentation](https://mmsegmentation.readthedocs.io/en/1.x/advanced_guides/structures.html) and [data element documentation](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html) in [MMEngine](https://github.com/open-mmlab/mmengine) for more information.

### Data Preprocessor to Model

Though drawn separately in the diagram [above](#overview-of-dataflow), data_preprocessor is a part of the model and thus can be found in [Model tutorial](./models.md) at Seg DataPreprocessor chapter.

The return value is the same as `PackSegInputs` except the `inputs` would be transferred to GPU and some additional metainfo like 'pad_shape' and 'padding_size' would be added to the `data_samples`.

### Model to Evaluator

At the evaluation procedure, the inference results would be transferred to `Evaluator`. You might read the [evaluation document](./evaluation.md) for more information about `Evaluator`.

After inference, the [BaseSegmentor](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/models/segmentors/base.py#L15) in MMSegmentation would do a simple post process to pack inference resutls, the segmentation logits produced by the neural network, segmentation mask after the `argmax` operation and ground truth(if exists) would be pack into a same SegDataSample instance. The return value of [postprocess_result](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/models/segmentors/base.py#L132) is a **`List` of `SegDataSample`**. Following diagram shows the key properties of these SegDataSample instances.

![SegDataSample](../../../resources/SegDataSample.png)

### Model to Loss function

The same as Data Preprocessor, loss function is also a part of the model, it's a property of [decode head](<>).

### Loss function to Optimizer
