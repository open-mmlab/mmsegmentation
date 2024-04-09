import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample

import os
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
from AI4ArcticSeaIceChallenge.functions import chart_cbar


@HOOKS.register_module()
class SegAI4ArcticVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 downsample_factor=5,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.downsample_factor = downsample_factor
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
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')
        self._test_index = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        window_name = f'val_{osp.basename(img_path)}'

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[SegDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.img_path
            window_name = f'test_{osp.basename(img_path)}'
            xarr = xr.open_dataset(img_path, engine='h5netcdf')
            img = xarr[['nersc_sar_primary',
                        'nersc_sar_secondary']].to_array().data
            img = np.transpose(img, (1, 2, 0))
            shape = img.shape
            scene_name = os.path.basename(img_path)
            scene_name = scene_name[17:32] + '_' + \
                scene_name[77:80] + '_prep.nc'
            if self.downsample_factor != 1:
                img = torch.from_numpy(img)
                img = img.unsqueeze(0).permute(0, 3, 1, 2)
                img = torch.nn.functional.interpolate(img,
                                                      size=(shape[0]//self.downsample_factor,
                                                            shape[1]//self.downsample_factor),
                                                      mode='nearest')
                img = img.permute(0, 2, 3, 1).squeeze(0)
                img = img.numpy()
            os.makedirs(osp.join(runner.cfg.work_dir, 'test'), exist_ok=True)
            fig, axs2d = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))

            axs = axs2d.flat

            # HH HV
            for j in range(0, 2):
                ax = axs[j]
                if j == 0:
                    ax.set_title(f'Scene {scene_name}, HH')
                else:
                    ax.set_title(f'Scene {scene_name}, HV')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(img[:, :, j], cmap='gray')

            gt_sem_seg = data_sample.gt_sem_seg.data.cpu().numpy()[
                0, :, :].astype(np.float16)
            gt_sem_seg[gt_sem_seg == 255] = np.nan

            pred_sem_seg = data_sample.pred_sem_seg.data.cpu().numpy()[
                0, :, :].astype(np.float16)
            nan_mask = np.isnan(gt_sem_seg)
            pred_sem_seg[nan_mask] = np.nan
            # GT
            ax = axs[2]
            ax.imshow(gt_sem_seg, vmin=0, vmax=5,
                      cmap='jet', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Scene {scene_name}, SOD: Ground Truth')
            chart_cbar(ax=ax, n_classes=7, chart='SOD', cmap='jet')
            # HH without land
            ax = axs[3]
            HH = img[:,:,0].copy()
            HH[nan_mask] = np.nan
            ax.imshow(HH, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Scene {scene_name}, HH without land')
            # chart_cbar(ax=ax, n_classes=7, chart='SOD', cmap='jet')
            # HV without land
            ax = axs[4]
            HV = img[:,:,0].copy()
            HV[nan_mask] = np.nan
            ax.imshow(HV, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Scene {scene_name}, HV without land')
            # chart_cbar(ax=ax, n_classes=7, chart='SOD', cmap='jet')

            ax = axs[5]
            ax.imshow(pred_sem_seg, vmin=0, vmax=5,
                      cmap='jet', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Scene {scene_name}, SOD: Pred')
            chart_cbar(ax=ax, n_classes=7, chart='SOD', cmap='jet')


            plt.subplots_adjust(left=0, bottom=0, right=1,
                                top=0.75, wspace=0.5, hspace=-0)
            fig.savefig(f"{osp.join(runner.cfg.work_dir,'test', scene_name)}.png",
                        format='png', dpi=128, bbox_inches="tight")
            plt.close('all')
