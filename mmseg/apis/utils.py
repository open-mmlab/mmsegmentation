# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Sequence, Union

import numpy as np
from mmengine.dataset import Compose
from mmengine.model import BaseModel

ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def _preprare_data(imgs: ImageType, model: BaseModel):

    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') == 'LoadAnnotations':
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

    return data, is_batch
