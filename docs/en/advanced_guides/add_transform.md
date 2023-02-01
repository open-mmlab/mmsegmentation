# Adding New Data Transforms

1. Write a new pipeline in any file, e.g., `my_pipeline.py`. It takes a dict as input and return a dict.

   ```python
   from mmseg.datasets import TRANSFORMS
   @TRANSFORMS.register_module()
   class MyTransform:
       def transform(self, results):
           results['dummy'] = True
           return results
   ```

2. Import the new class.

   ```python
   from .my_pipeline import MyTransform
   ```

3. Use it in config files.

   ```python
   crop_size = (512, 1024)
   train_pipeline = [
       dict(type='LoadImageFromFile'),
       dict(type='LoadAnnotations'),
       dict(type='RandomResize',
            scale=(2048, 1024),
            ratio_range=(0.5, 2.0),
            keep_ratio=True),
       dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
       dict(type='RandomFlip', flip_ratio=0.5),
       dict(type='PhotoMetricDistortion'),
       dict(type='MyTransform'),
       dict(type='PackSegInputs'),
   ]
   ```
