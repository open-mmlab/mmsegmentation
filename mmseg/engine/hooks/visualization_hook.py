# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class SegVisualizationHook(Hook):
    """Segmentation Visualization Hook. This hook visualize the prediction
    results during validation and testing.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file client.
            Defaults to None.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')
            self.wait_time = wait_time
        else:
            self.wait_time = 0.
        self.draw = draw
        self.backend_args = backend_args

    def _create_dummy_image(self, height: int, width: int) -> np.ndarray:
        """Create a dummy black image for visualization.

        Args:
            height: Image height
            width: Image width

        Returns:
            Black RGB image of shape (H, W, 3)
        """
        return np.zeros((height, width, 3), dtype=np.uint8)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
        """
        if self.draw is False:
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_name = osp.basename(img_path)

                # Get image size from prediction
                if hasattr(output, 'pred_sem_seg'):
                    pred_shape = output.pred_sem_seg.data.shape
                    h, w = pred_shape[-2], pred_shape[-1]
                elif hasattr(output, 'gt_sem_seg'):
                    gt_shape = output.gt_sem_seg.data.shape
                    h, w = gt_shape[-2], gt_shape[-1]
                else:
                    # Fallback to default size
                    h, w = 512, 512

                # Create dummy image instead of loading original
                img = self._create_dummy_image(h, w)

                # Only draw prediction, not ground truth
                self._visualizer.add_datasample(
                    img_name,
                    img,
                    data_sample=output,
                    draw_gt=False,  # Don't draw GT
                    draw_pred=True,  # Only draw prediction
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter,
                    with_labels=False)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[SegDataSample]) -> None:
        """Run after every testing iteration.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]]): Outputs from model.
        """
        if self.draw is False:
            return

        for output in outputs:
            img_path = output.img_path
            img_name = osp.basename(img_path)

            # Get image size from prediction
            if hasattr(output, 'pred_sem_seg'):
                pred_shape = output.pred_sem_seg.data.shape
                h, w = pred_shape[-2], pred_shape[-1]
            elif hasattr(output, 'gt_sem_seg'):
                gt_shape = output.gt_sem_seg.data.shape
                h, w = gt_shape[-2], gt_shape[-1]
            else:
                # Fallback to default size
                h, w = 512, 512

            # Create dummy image instead of loading original
            img = self._create_dummy_image(h, w)

            # 关键修改：不指定 out_file，让 visualizer 使用 --show-dir 的设置
            # Only draw prediction, not ground truth
            self._visualizer.add_datasample(
                img_name,
                img,
                data_sample=output,
                draw_gt=False,  # Don't draw GT
                draw_pred=True,  # Only draw prediction
                show=self.show,
                wait_time=self.wait_time,
                step=runner.iter,
                with_labels=False)  # 移除了 out_file 参数