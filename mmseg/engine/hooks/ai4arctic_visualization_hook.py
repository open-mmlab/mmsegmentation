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
from mmengine.dist.utils import master_only
from torchmetrics.functional import r2_score, f1_score, mean_squared_error
import wandb


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
                 device: str = 'cuda:0',
                 metrics: list[dict] = {
                     'SIC': 'r2', 'SOD': 'f1', 'FLOE': 'f1'},
                 num_classes: dict = None,
                 pad_size=None,
                 nan=255,
                 backend_args: Optional[dict] = None,
                 tasks=[''],
                 combined_score_weights=[2, 2, 1]):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.downsample_factor = downsample_factor
        self.interval = interval
        self.show = show
        self.device = device
        self.GT_flat = torch.Tensor().to(device)
        self.pred_flat = torch.Tensor().to(device)
        self.metrics = metrics
        self.num_classes = num_classes
        self.pad_size = pad_size
        self.nan = nan
        self.tasks = tasks
        self.combined_score_weights = combined_score_weights
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

    def before_val(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before validation.

        Args:
            runner (Runner): The runner of the validation process.
        """
        self.GT_flat = {}
        self.pred_flat = {}

        for task in self.tasks:
            self.GT_flat[task] = torch.Tensor().to(self.device)
            self.pred_flat[task] = torch.Tensor().to(self.device)

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
                                                      size=(shape[0] // self.downsample_factor,
                                                            shape[1] // self.downsample_factor),
                                                      mode='nearest')
                img = img.permute(0, 2, 3, 1).squeeze(0)
                if self.pad_size is not None:
                    # Calculate the pad amounts
                    pad_height = max(0, self.pad_size[0] - img.shape[0])
                    pad_width = max(0, self.pad_size[1] - img.shape[1])
                    # Pad the image
                    img = torch.nn.functional.pad(
                        img, (0, 0, 0, pad_width, 0, pad_height), mode='constant', value=self.nan)
                img = img.numpy()
            os.makedirs(osp.join(runner.cfg.work_dir, 'val'), exist_ok=True)
            fig, axs2d = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))

            axs = axs2d.flat

            # # HH HV
            # for j in range(0, 2):
            #     ax = axs[j]
            #     if j == 0:
            #         ax.set_title(f'Scene {scene_name}, HH')
            #     else:
            #         ax.set_title(f'Scene {scene_name}, HV')
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            #     ax.imshow(img[:, :, j], cmap='gray')

            # Ground Truth and Predictions for SIC, SOD, FLOE
            tasks = self.tasks
            gt_sem_seg = {}
            for i, task in enumerate(tasks):
                gt_sem_seg = data_sample.gt_sem_seg.data.cpu().numpy()[
                    i, :, :].astype(np.float16)
                gt_sem_seg[gt_sem_seg == 255] = np.nan

                pred_sem_seg = data_sample.get(f'pred_sem_seg_{task}').data.cpu().numpy()[
                    0, :, :].astype(np.float16)
                nan_mask = np.isnan(gt_sem_seg)
                pred_sem_seg[nan_mask] = np.nan

                # Ground Truth
                ax = axs[3 + i]
                ax.imshow(
                    gt_sem_seg, vmin=0, vmax=self.num_classes[task] - 1, cmap='jet', interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Scene {scene_name}, {task}: Ground Truth')
                chart_cbar(
                    ax=ax, n_classes=self.num_classes[task], chart=task, cmap='jet')

                # Predictions
                ax = axs[6 + i]
                ax.imshow(pred_sem_seg, vmin=0,
                          vmax=self.num_classes[task] - 1, cmap='jet', interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Scene {scene_name}, {task}: Pred')
                chart_cbar(
                    ax=ax, n_classes=self.num_classes[task], chart=task, cmap='jet')

            # HH without land
            ax = axs[0]
            HH = img[:, :, 0].copy()
            HH[nan_mask] = np.nan
            ax.imshow(HH, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Scene {scene_name}, HH without land')

            # HV without land
            ax = axs[1]
            HV = img[:, :, 1].copy()
            HV[nan_mask] = np.nan
            ax.imshow(HV, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Scene {scene_name}, HV without land')

            plt.subplots_adjust(left=0, bottom=0, right=1,
                                top=0.75, wspace=0.5, hspace=-0)
            fig.savefig(f"{osp.join(runner.cfg.work_dir, 'val', scene_name)}.png",
                        format='png', dpi=128, bbox_inches="tight")
            plt.close('all')

            for i, task in enumerate(tasks):
                gt_sem_seg = data_sample.gt_sem_seg.data.cpu().numpy()[
                    i, :, :].astype(np.float16)
                gt_sem_seg[gt_sem_seg == 255] = np.nan
                pred_sem_seg = data_sample.get(f'pred_sem_seg_{task}').data.cpu().numpy()[
                    0, :, :].astype(np.float16)
                nan_mask = np.isnan(gt_sem_seg)

                self.GT_flat[task] = torch.cat((self.GT_flat[task], torch.tensor(
                    gt_sem_seg[~nan_mask], device=self.device)))
                self.pred_flat[task] = torch.cat((self.pred_flat[task], torch.tensor(
                    pred_sem_seg[~nan_mask], device=self.device)))

    @master_only
    def after_val(self, runner) -> None:
        """Calculate R2 score between flattened GT and flattend pred

        Args:
            runner (Runner): The runner of the testing process.
        """
        combined_score = 0
        for i, task in enumerate(self.tasks):
            if self.metrics[task] == 'r2':
                r2 = r2_score(
                    preds=self.pred_flat[task], target=self.GT_flat[task])
                combined_score += self.combined_score_weights[i]*r2.item()
                print(f'R2 score: [{task}]', r2.item())
                runner.visualizer._vis_backends['WandbVisBackend']._commit = False
                runner.visualizer._vis_backends['WandbVisBackend']._wandb.log(
                    {f'val/R2 score [{task}]': r2.item()})
                runner.visualizer._vis_backends['WandbVisBackend']._commit = True
                # wandb.summary["R2 score"] = r2
                # wandb.save()
            elif self.metrics[task] == 'f1':
                f1 = f1_score(target=self.GT_flat[task], preds=self.pred_flat[task],
                              average='weighted', task='multiclass', num_classes=self.num_classes[task])
                combined_score += self.combined_score_weights[i]*f1.item()
                print(f'F1 score: [{task}]', f1.item())
                runner.visualizer._vis_backends['WandbVisBackend']._commit = False
                runner.visualizer._vis_backends['WandbVisBackend']._wandb.log(
                    {f'val/F1 score [{task}]': f1.item()})
                runner.visualizer._vis_backends['WandbVisBackend']._commit = True
                # wandb.summary["F1 score"] = f1.item()
                # wandb.save(runner.cfg.filename)
            else:
                raise "Unrecognized metric"
        print(f'Combined score:', combined_score)
        runner.visualizer._vis_backends['WandbVisBackend']._commit = False
        runner.visualizer._vis_backends['WandbVisBackend']._wandb.log(
            {f'val/Combined score': combined_score})
        runner.visualizer._vis_backends['WandbVisBackend']._commit = True

    def before_test(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before testing.

        Args:
            runner (Runner): The runner of the testing process.
        """

        self.GT_flat = {}
        self.pred_flat = {}

        for task in self.tasks:
            self.GT_flat[task] = torch.Tensor().to(self.device)
            self.pred_flat[task] = torch.Tensor().to(self.device)

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
                                                      size=(shape[0] // self.downsample_factor,
                                                            shape[1] // self.downsample_factor),
                                                      mode='nearest')
                img = img.permute(0, 2, 3, 1).squeeze(0)
                if self.pad_size is not None:
                    # Calculate the pad amounts
                    pad_height = max(0, self.pad_size[0] - img.shape[0])
                    pad_width = max(0, self.pad_size[1] - img.shape[1])
                    # Pad the image
                    img = torch.nn.functional.pad(
                        img, (0, 0, 0, pad_width, 0, pad_height), mode='constant', value=self.nan)
                img = img.numpy()
            os.makedirs(osp.join(runner.cfg.work_dir, 'test'), exist_ok=True)
            fig, axs2d = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))

            axs = axs2d.flat

            # # HH HV
            # for j in range(0, 2):
            #     ax = axs[j]
            #     if j == 0:
            #         ax.set_title(f'Scene {scene_name}, HH')
            #     else:
            #         ax.set_title(f'Scene {scene_name}, HV')
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            #     ax.imshow(img[:, :, j], cmap='gray')

            # Ground Truth and Predictions for SIC, SOD, FLOE
            tasks = self.tasks
            gt_sem_seg = {}
            for i, task in enumerate(tasks):
                gt_sem_seg = data_sample.gt_sem_seg.data.cpu().numpy()[
                    i, :, :].astype(np.float16)
                gt_sem_seg[gt_sem_seg == 255] = np.nan

                pred_sem_seg = data_sample.get(f'pred_sem_seg_{task}').data.cpu().numpy()[
                    0, :, :].astype(np.float16)
                nan_mask = np.isnan(gt_sem_seg)
                pred_sem_seg[nan_mask] = np.nan

                # Ground Truth
                ax = axs[3 + i]
                ax.imshow(
                    gt_sem_seg, vmin=0, vmax=self.num_classes[task] - 1, cmap='jet', interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Scene {scene_name}, {task}: Ground Truth')
                chart_cbar(
                    ax=ax, n_classes=self.num_classes[task], chart=task, cmap='jet')

                # Predictions
                ax = axs[6 + i]
                ax.imshow(pred_sem_seg, vmin=0,
                          vmax=self.num_classes[task] - 1, cmap='jet', interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Scene {scene_name}, {task}: Pred')
                chart_cbar(
                    ax=ax, n_classes=self.num_classes[task], chart=task, cmap='jet')

            # HH without land
            ax = axs[0]
            HH = img[:, :, 0].copy()
            HH[nan_mask] = np.nan
            ax.imshow(HH, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Scene {scene_name}, HH without land')

            # HV without land
            ax = axs[1]
            HV = img[:, :, 1].copy()
            HV[nan_mask] = np.nan
            ax.imshow(HV, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Scene {scene_name}, HV without land')

            plt.subplots_adjust(left=0, bottom=0, right=1,
                                top=0.75, wspace=0.5, hspace=-0)
            fig.savefig(f"{osp.join(runner.cfg.work_dir, 'test', scene_name)}.png",
                        format='png', dpi=128, bbox_inches="tight")
            plt.close('all')

            for i, task in enumerate(tasks):
                gt_sem_seg = data_sample.gt_sem_seg.data.cpu().numpy()[
                    i, :, :].astype(np.float16)
                gt_sem_seg[gt_sem_seg == 255] = np.nan
                pred_sem_seg = data_sample.get(f'pred_sem_seg_{task}').data.cpu().numpy()[
                    0, :, :].astype(np.float16)
                nan_mask = np.isnan(gt_sem_seg)

                self.GT_flat[task] = torch.cat((self.GT_flat[task], torch.tensor(
                    gt_sem_seg[~nan_mask], device=self.device)))
                self.pred_flat[task] = torch.cat((self.pred_flat[task], torch.tensor(
                    pred_sem_seg[~nan_mask], device=self.device)))

    @master_only
    def after_test(self, runner) -> None:
        """Calculate R2 score between flattened GT and flattend pred

        Args:
            runner (Runner): The runner of the testing process.
        """
        combined_score = 0
        for i, task in enumerate(self.tasks):
            if self.metrics[task] == 'r2':
                r2 = r2_score(
                    preds=self.pred_flat[task], target=self.GT_flat[task])
                combined_score += self.combined_score_weights[i]*r2.item()
                print(f'R2 score: [{task}]', r2.item())
                runner.visualizer._vis_backends['WandbVisBackend']._commit = False
                runner.visualizer._vis_backends['WandbVisBackend']._wandb.log(
                    {f'test/R2 score [{task}]': r2.item()})
                runner.visualizer._vis_backends['WandbVisBackend']._commit = True
                # wandb.summary["R2 score"] = r2
                # wandb.save()
            elif self.metrics[task] == 'f1':
                f1 = f1_score(target=self.GT_flat[task], preds=self.pred_flat[task],
                              average='weighted', task='multiclass', num_classes=self.num_classes[task])
                combined_score += self.combined_score_weights[i]*f1.item()
                print(f'F1 score: [{task}]', f1.item())
                runner.visualizer._vis_backends['WandbVisBackend']._commit = False
                runner.visualizer._vis_backends['WandbVisBackend']._wandb.log(
                    {f'test/F1 score [{task}]': f1.item()})
                runner.visualizer._vis_backends['WandbVisBackend']._commit = True
                # wandb.summary["F1 score"] = f1.item()
                # wandb.save(runner.cfg.filename)
            else:
                raise "Unrecognized metric"
        print(f'Combined score:', combined_score)
        runner.visualizer._vis_backends['WandbVisBackend']._commit = False
        runner.visualizer._vis_backends['WandbVisBackend']._wandb.log(
            {f'test/Combined score': combined_score})
        runner.visualizer._vis_backends['WandbVisBackend']._commit = True
