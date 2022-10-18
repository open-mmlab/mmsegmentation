# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class PackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)

        data_sample = SegDataSample()

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                data=to_tensor(results['gt_seg_map'][None,
                                                     ...].astype(np.int64)))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_labels' in results and 'gt_masks' in results:
            instance_data = InstanceData()
            instance_data['det_labels'] = to_tensor(results['gt_labels'])
            instance_data['masks'] = to_tensor(results['gt_masks'])
            data_sample.gt_instances = instance_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class GenerateInstanceLabel:
    """Transfer gt_semantic_seg to binary mask and generate gt_labels."""

    def __init__(self, ignore_index: int = 255):
        self.ignore_index = ignore_index

    def transform(self, results: dict) -> dict:
        gt_semantic_seg = results['gt_seg_map']
        gt_labels = np.unique(gt_semantic_seg)
        # remove ignored region
        gt_labels = gt_labels[gt_labels != self.ignore_index]

        gt_masks = []
        for class_id in gt_labels:
            gt_masks.append(gt_semantic_seg == class_id)

        if len(gt_masks) == 0:
            # Some image does not have annotation (all ignored)
            gt_masks = np.empty(
                (0, ) + results['pad_shape'][:-1], dtype=np.int64)
            gt_labels = np.empty((0, ), dtype=np.int64)
        else:
            gt_masks = np.asarray(gt_masks, dtype=np.int64)
            gt_labels = np.asarray(gt_labels, dtype=np.int64)

        results['gt_labels'] = gt_labels
        results['gt_masks'] = gt_masks
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(ignore_index={self.ignore_index})'
