# 新增数据增强

## 自定义数据增强

自定义数据增强必须继承 `BaseTransform` 并实现 `transform` 函数。这里我们使用一个简单的翻转变换作为示例：

```python
import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class MyFlip(BaseTransform):
    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = mmcv.imflip(img, direction=self.direction)
        return results
```

此外，新的类需要被导入。

```python
from .my_pipeline import MyFlip
```

这样，我们就可以实例化一个 `MyFlip` 对象并使用它来处理数据字典。

```python
import numpy as np

transform = MyFlip(direction='horizontal')
data_dict = {'img': np.random.rand(224, 224, 3)}
data_dict = transform(data_dict)
processed_img = data_dict['img']
```

或者，我们可以在配置文件中的数据流程中使用 `MyFlip` 变换。

```python
pipeline = [
    ...
    dict(type='MyFlip', direction='horizontal'),
    ...
]
```

需要注意，如果要在配置文件中使用 `MyFlip`，必须确保在运行时导入了包含 `MyFlip` 的文件。
