# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Optional, Sequence, Union

import mmcv
import mmengine
import numpy as np
from mmcv.transforms import Compose
from mmengine.infer.infer import BaseInferencer, ModelType

from mmseg.structures import SegDataSample
from mmseg.utils import ConfigType, SampleList, register_all_modules
from mmseg.visualization import SegLocalVisualizer

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[SegDataSample, SampleList]


class MMSegInferencer(BaseInferencer):
    """MMSegInferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "fcn_r50-d8_4xb2-40k_cityscapes-512x1024" or
            "configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py"
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
    """

    preprocess_kwargs: set = set()
    forward_kwargs: set = {'mode'}
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'draw_pred', 'img_out_dir',
        'opacity'
    }
    postprocess_kwargs: set = {
        'print_result', 'pred_out_file', 'return_datasample', 'save_mask'
    }

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 palette: Optional[Union[str, List]] = None,
                 classes: Optional[Union[str, List]] = None,
                 dataset_name: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmseg') -> None:
        # A global counter tracking the number of images processes, for
        # naming of the output images
        self.num_visualized_imgs = 0
        register_all_modules()
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)

        assert isinstance(self.visualizer, SegLocalVisualizer)
        self.visualizer.set_dataset_meta(palette, classes, dataset_name)

    def __call__(self,
                 inputs: InputsType,
                 mode: str = 'predict',
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 return_vis: bool = False,
                 show: bool = False,
                 wait_time: int = 0,
                 draw_pred: bool = True,
                 img_out_dir: str = '',
                 print_result: bool = False,
                 pred_out_file: str = '',
                 save_mask: bool = False,
                 **kwargs) -> dict:
        return super().__call__(
            inputs=inputs,
            return_datasamples=return_datasamples,
            batch_size=batch_size,
            mode=mode,
            return_vis=return_vis,
            show=show,
            wait_time=wait_time,
            draw_pred=draw_pred,
            img_out_dir=img_out_dir,
            print_result=print_result,
            pred_out_file=pred_out_file,
            save_mask=save_mask,
            **kwargs)

    def visualize(self,
                  inputs: list,
                  preds: List[dict],
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  img_out_dir: str = '',
                  opacity: float = 0.8) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (list): Inputs preprocessed by :meth:`_inputs_to_list`.
            preds (Any): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.

        Returns:
            List[np.ndarray]: Visualization results.
        """
        if self.visualizer is None or (not show and img_out_dir == ''
                                       and not return_vis):
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None')

        self.visualizer.set_dataset_meta(
            classes=self.model.dataset_meta.get('classes', None),
            palette=self.model.dataset_meta.get('palette', None))
        self.visualizer.alpha = opacity

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type:'
                                 f'{type(single_input)}')

            out_file = osp.join(img_out_dir, img_name) if img_out_dir != ''\
                else None

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                out_file=out_file)
            results.append(img)
            self.num_visualized_imgs += 1

        return results

    def postprocess(self,
                    preds: PredType,
                    visualization: List[np.ndarray],
                    return_datasample: bool = False,
                    mask_dir: str = 'mask',
                    save_mask: bool = True,
                    print_result: bool = False,
                    pred_out_file: str = '') -> dict:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (np.ndarray): Visualized predictions.
            return_datasample (bool): Whether to return results as datasamples.
                Defaults to False.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            pred_out_file: File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``

            - ``visualization (Any)``: Returned by :meth:`visualize`
            - ``predictions`` (dict or DataSample): Returned by
              :meth:`forward` and processed in :meth:`postprocess`.
              If ``return_datasample=False``, it usually should be a
              json-serializable dict containing only basic data elements such
              as strings and numbers.
        """
        results_dict = {}

        results_dict['predictions'] = preds
        results_dict['visualization'] = visualization

        if print_result:
            print_result(results_dict)
        if pred_out_file != '':
            if save_mask:
                mmengine.mkdir_or_exist(pred_out_file)
                preds = [preds] if isinstance(preds, SegDataSample) else preds
                for pred in preds:
                    mmcv.imwrite(
                        pred.pred_sem_seg.numpy().data[0],
                        osp.join(pred_out_file, mask_dir,
                                 osp.basename(pred.metainfo['img_path'])))
            else:
                mmengine.dump(results_dict, pred_out_file)

        if return_datasample:
            return preds

        return results_dict

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline.

        Return a pipeline to handle various input data, such as ``str``,
        ``np.ndarray``. It is an abstract method in BaseInferencer, and should
        be implemented in subclasses.

        The returned pipeline will be used to process a single data.
        It will be used in :meth:`preprocess` like this:

        .. code-block:: python
            def preprocess(self, inputs, batch_size, **kwargs):
                ...
                dataset = map(self.pipeline, dataset)
                ...
        """
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        # Loading annotations is also not applicable
        idx = self._get_transform_idx(pipeline_cfg, 'LoadAnnotations')
        if idx != -1:
            del pipeline_cfg[idx]
        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFile')

        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'InferencerLoader'
        return Compose(pipeline_cfg)

    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] == name:
                return i
        return -1
