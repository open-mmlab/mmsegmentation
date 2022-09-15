# Overview

This chapter introduces you to the framework of MMSegmentation, the basic conception of semantic segmentation, and provides links to detailed tutorials about MMSegmentation.

## What is semantic segmentation?

Semantic segmentation is the task of clustering parts of an image together that belong to the same object class.
It is a form of pixel-level prediction because each pixel in an image is classified according to a category.
Some example benchmarks for this task are [Cityscapes](https://www.cityscapes-dataset.com/benchmarks/), [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/).
Models are usually evaluated with the Mean Intersection-Over-Union (Mean IoU) and Pixel Accuracy metrics.

## What is MMSegmentation?

MMSegmentation is a toolbox that provides a framework for unified implementation and evaluation of semant
ic segmentation methods,
and contains high-quality implementations of popular semantic segmentation methods and datasets.

MMSeg consists of 7 main parts including apis, structures, datasets, models, engine, evaluation and visualization.

- **apis** provides high-level APIs for model inference.

- **structures** provides segmentation data structure SegDataSample.

- **datasets** supports various dataset for semantic segmentation.

  - **transforms** contains a lot of useful data augmentation transforms.

- **models** is the most vital part for segmentors and contains different components of a segmentor.

  - **segmentor** defines all of the segmentation model classes.
  - **data_preprocessors** is for preprocessing the input data of the model.
  - **backbones** contains various backbone networks that transform an image to feature maps.
  - **necks** contains various neck components that connect the backbone and heads.
  - **decode_heads** contains various head components that take feature map as input and predict segmentation results.
  - **losses** contains various loss functions

- **engine** is a part for runtime components that extends function of [MMEngine](https://github.com/open-mmlab/mmengine).

  - **optimizers** provides optimizers and optimizer wrappers.
  - **hooks** provides various hooks of the runner.

- **evaluation** provides different metrics for evaluating model performance.

- **visualization** is for visualizing segmentation results.

## How to use this documentation

Here is a detailed step-by-step guide to learning more about MMSegmentation:

1. For installation instructions, please see [get_started](getting_started.md).

2. Refer to the tutorials below for the basic usage of MMSegmentation:

   - [Config](user_guides/1_config.md)
   - [Dataset preparation](user_guides/2_dataset_prepare.md)
   - [Inference](user_guides/3_inference.md)
   - [Train and Test](user_guides/4_train_test.md)

3. Refer to the tutorials below to dive deeper:

   - [Data flow](advanced_guides/data_flow.md)
   - [Structures](advanced_guides/structures.md)
   - [Models](advanced_guides/models.md)
   - [Datasets](advanced_guides/models.md)
   - [Evaluation](advanced_guides/evaluation.md)

4. Refer to tutorials below to build your own segmentation project:

   - [Add a new model](advanced_guides/add_models.md)
   - [Add a new datasets](advanced_guides/add_dataset.md)
   - [Add new transforms](advanced_guides/add_transform.md)
   - [Custormize runtime](advanced_guides/customize_runtime.md)

## References

- https://paperswithcode.com/task/semantic-segmentation/codeless#task-home
