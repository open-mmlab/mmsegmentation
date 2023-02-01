# Overview

This chapter introduces you to the framework of MMSegmentation, and the basic conception of semantic segmentation. It also provides links to detailed tutorials about MMSegmentation.

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

- **structures** provides segmentation data structure `SegDataSample`.

- **datasets** supports various datasets for semantic segmentation.

  - **transforms** contains a lot of useful data augmentation transforms.

- **models** is the most vital part for segmentors and contains different components of a segmentor.

  - **segmentors** defines all of the segmentation model classes.
  - **data_preprocessors** works for preprocessing the input data of the model.
  - **backbones** contains various backbone networks that transform an image to feature maps.
  - **necks** contains various neck components that connect the backbone and heads.
  - **decode_heads** contains various head components that take feature map as input and predict segmentation results.
  - **losses** contains various loss functions.

- **engine** is a part for runtime components that extends function of [MMEngine](https://github.com/open-mmlab/mmengine).

  - **optimizers** provides optimizers and optimizer wrappers.
  - **hooks** provides various hooks of the runner.

- **evaluation** provides different metrics for evaluating model performance.

- **visualization** is for visualizing segmentation results.

## How to use this documentation

Here is a detailed step-by-step guide to learn more about MMSegmentation:

1. For installation instructions, please see [get_started](getting_started.md).

2. For beginners, MMSegmentation is the best place to start the journey of semantic segmentation
   as there are many SOTA and classic segmentation [models](model_zoo.md),
   and it is easier to carry out a segmentation task by plugging together building blocks and convenient high-level apis.
   Refer to the tutorials below for the basic usage of MMSegmentation:

   - [Config](user_guides/1_config.md)
   - [Dataset Preparation](user_guides/2_dataset_prepare.md)
   - [Inference](user_guides/3_inference.md)
   - [Train and Test](user_guides/4_train_test.md)

3. If you would like to learn about the fundamental classes and features that make MMSegmentation work,
   please refer to the tutorials below to dive deeper:

   - [Data flow](advanced_guides/data_flow.md)
   - [Structures](advanced_guides/structures.md)
   - [Models](advanced_guides/models.md)
   - [Datasets](advanced_guides/datasets.md)
   - [Evaluation](advanced_guides/evaluation.md)

4. MMSegmentation also provide tutorials for customization and advanced research,
   please refer to the below guides to build your own segmentation project:

   - [Add new models](advanced_guides/add_models.md)
   - [Add new datasets](advanced_guides/add_dataset.md)
   - [Add new transforms](advanced_guides/add_transform.md)
   - [Customize runtime](advanced_guides/customize_runtime.md)

5. If you are more familiar with MMSegmentation v0.x, there is documentation about migration from MMSegmentation v0.x to v1.x

   - [migration](migration/index.rst)

## References

- https://paperswithcode.com/task/semantic-segmentation/codeless#task-home
