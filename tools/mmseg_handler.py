# Copyright (c) OpenMMLab. All rights reserved.
import base64
import io
import os

import cv2
import mmcv
import torch
from ts.torch_handler.base_handler import BaseHandler

from mmseg.apis import inference_segmentor, init_segmentor


class MMsegHandler(BaseHandler):

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        self.model = init_segmentor(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data):
        images = []

        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)

        return images

    def inference(self, data, *args, **kwargs):
        results = [inference_segmentor(self.model, img) for img in data]
        return results

    def postprocess(self, data):
        output = []
        for image_result in data:
            buffer = io.BytesIO()
            _, buffer = cv2.imencode('.png', image_result[0].astype('uint8'))
            output.append(buffer.tobytes())
        return output
