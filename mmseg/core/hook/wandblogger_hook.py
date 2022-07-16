# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook

from mmseg.core import DistEvalHook, EvalHook


@HOOKS.register_module()
class MMSegWandbHook(WandbLoggerHook):
    """Enhanced Wandb logger hook for MMSegmentation.

    Comparing with the :cls:`mmcv.runner.WandbLoggerHook`, this hook can not
    only automatically log all the metrics but also log the following extra
    information - saves model checkpoints as W&B Artifact, and
    logs model prediction as interactive W&B Tables.

    - Metrics: The MMSegWandbHook will automatically log training
      and validation metrics along with system metrics (CPU/GPU).

    - Checkpointing: If `log_checkpoint` is True, the checkpoint saved at
      every checkpoint interval will be saved as W&B Artifacts.
      This depends on the : class:`mmcv.runner.CheckpointHook` whose priority
      is higher than this hook. Please refer to
      https://docs.wandb.ai/guides/artifacts/model-versioning
      to learn more about model versioning with W&B Artifacts.

    - Checkpoint Metadata: If evaluation results are available for a given
      checkpoint artifact, it will have a metadata associated with it.
      The metadata contains the evaluation metrics computed on validation
      data with that checkpoint along with the current epoch. It depends
      on `EvalHook` whose priority is more than MMSegWandbHook.

    - Evaluation: At every evaluation interval, the `MMSegWandbHook` logs the
      model prediction as interactive W&B Tables. The number of samples
      logged is given by `num_eval_images`. Currently, the `MMSegWandbHook`
      logs the predicted segmentation masks along with the ground truth at
      every evaluation interval. This depends on the `EvalHook` whose
      priority is more than `MMSegWandbHook`. Also note that the data is just
      logged once and subsequent evaluation tables uses reference to the
      logged data to save memory usage. Please refer to
      https://docs.wandb.ai/guides/data-vis to learn more about W&B Tables.

    ```
    Example:
        log_config = dict(
            ...
            hooks=[
                ...,
                dict(type='MMSegWandbHook',
                     init_kwargs={
                         'entity': "YOUR_ENTITY",
                         'project': "YOUR_PROJECT_NAME"
                     },
                     interval=50,
                     log_checkpoint=True,
                     log_checkpoint_metadata=True,
                     num_eval_images=100,
                     bbox_score_thr=0.3)
            ])
    ```

    Args:
        init_kwargs (dict): A dict passed to wandb.init to initialize
            a W&B run. Please refer to https://docs.wandb.ai/ref/python/init
            for possible key-value pairs.
        interval (int): Logging interval (every k iterations).
            Default 10.
        log_checkpoint (bool): Save the checkpoint at every checkpoint interval
            as W&B Artifacts. Use this for model versioning where each version
            is a checkpoint.
            Default: False
        log_checkpoint_metadata (bool): Log the evaluation metrics computed
            on the validation data with the checkpoint, along with current
            epoch as a metadata to that checkpoint.
            Default: True
        num_eval_images (int): Number of validation images to be logged.
            Default: 100
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=50,
                 log_checkpoint=False,
                 log_checkpoint_metadata=False,
                 num_eval_images=100,
                 **kwargs):
        super(MMSegWandbHook, self).__init__(init_kwargs, interval, **kwargs)

        self.log_checkpoint = log_checkpoint
        self.log_checkpoint_metadata = (
            log_checkpoint and log_checkpoint_metadata)
        self.num_eval_images = num_eval_images
        self.log_evaluation = (num_eval_images > 0)
        self.ckpt_hook: CheckpointHook = None
        self.eval_hook: EvalHook = None
        self.test_fn = None

    @master_only
    def before_run(self, runner):
        super(MMSegWandbHook, self).before_run(runner)

        # Check if EvalHook and CheckpointHook are available.
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, EvalHook):
                from mmseg.apis import single_gpu_test
                self.eval_hook = hook
                self.test_fn = single_gpu_test
            if isinstance(hook, DistEvalHook):
                from mmseg.apis import multi_gpu_test
                self.eval_hook = hook
                self.test_fn = multi_gpu_test

        # Check conditions to log checkpoint
        if self.log_checkpoint:
            if self.ckpt_hook is None:
                self.log_checkpoint = False
                self.log_checkpoint_metadata = False
                runner.logger.warning(
                    'To log checkpoint in MMSegWandbHook, `CheckpointHook` is'
                    'required, please check hooks in the runner.')
            else:
                self.ckpt_interval = self.ckpt_hook.interval

        # Check conditions to log evaluation
        if self.log_evaluation or self.log_checkpoint_metadata:
            if self.eval_hook is None:
                self.log_evaluation = False
                self.log_checkpoint_metadata = False
                runner.logger.warning(
                    'To log evaluation or checkpoint metadata in '
                    'MMSegWandbHook, `EvalHook` or `DistEvalHook` in mmseg '
                    'is required, please check whether the validation '
                    'is enabled.')
            else:
                self.eval_interval = self.eval_hook.interval
                self.val_dataset = self.eval_hook.dataloader.dataset
                # Determine the number of samples to be logged.
                if self.num_eval_images > len(self.val_dataset):
                    self.num_eval_images = len(self.val_dataset)
                    runner.logger.warning(
                        f'The num_eval_images ({self.num_eval_images}) is '
                        'greater than the total number of validation samples '
                        f'({len(self.val_dataset)}). The complete validation '
                        'dataset will be logged.')

        # Check conditions to log checkpoint metadata
        if self.log_checkpoint_metadata:
            assert self.ckpt_interval % self.eval_interval == 0, \
                'To log checkpoint metadata in MMSegWandbHook, the interval ' \
                f'of checkpoint saving ({self.ckpt_interval}) should be ' \
                'divisible by the interval of evaluation ' \
                f'({self.eval_interval}).'

        # Initialize evaluation table
        if self.log_evaluation:
            # Initialize data table
            self._init_data_table()
            # Add data to the data table
            self._add_ground_truth(runner)
            # Log ground truth data
            self._log_data_table()

    @master_only
    def after_train_iter(self, runner):
        if self.get_mode(runner) == 'train':
            # An ugly patch. The iter-based eval hook will call the
            # `after_train_iter` method of all logger hooks before evaluation.
            # Use this trick to skip that call.
            # Don't call super method at first, it will clear the log_buffer
            return super(MMSegWandbHook, self).after_train_iter(runner)
        else:
            super(MMSegWandbHook, self).after_train_iter(runner)

        if self.by_epoch:
            return

        # Save checkpoint and metadata
        if (self.log_checkpoint
                and self.every_n_iters(runner, self.ckpt_interval)
                or (self.ckpt_hook.save_last and self.is_last_iter(runner))):
            if self.log_checkpoint_metadata and self.eval_hook:
                metadata = {
                    'iter': runner.iter + 1,
                    **self._get_eval_results()
                }
            else:
                metadata = None
            aliases = [f'iter_{runner.iter+1}', 'latest']
            model_path = osp.join(self.ckpt_hook.out_dir,
                                  f'iter_{runner.iter+1}.pth')
            self._log_ckpt_as_artifact(model_path, aliases, metadata)

        # Save prediction table
        if self.log_evaluation and self.eval_hook._should_evaluate(runner):
            # Currently the results of eval_hook is not reused by wandb, so
            # wandb will run evaluation again internally. We will consider
            # refactoring this function afterwards
            results = self.test_fn(runner.model, self.eval_hook.dataloader)
            # Initialize evaluation table
            self._init_pred_table()
            # Log predictions
            self._log_predictions(results, runner)
            # Log the table
            self._log_eval_table(runner.iter + 1)

    @master_only
    def after_run(self, runner):
        self.wandb.finish()

    def _log_ckpt_as_artifact(self, model_path, aliases, metadata=None):
        """Log model checkpoint as  W&B Artifact.

        Args:
            model_path (str): Path of the checkpoint to log.
            aliases (list): List of the aliases associated with this artifact.
            metadata (dict, optional): Metadata associated with this artifact.
        """
        model_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(model_path)
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _get_eval_results(self):
        """Get model evaluation results."""
        results = self.eval_hook.latest_results
        eval_results = self.val_dataset.evaluate(
            results, logger='silent', **self.eval_hook.eval_kwargs)
        return eval_results

    def _init_data_table(self):
        """Initialize the W&B Tables for validation data."""
        columns = ['image_name', 'image']
        self.data_table = self.wandb.Table(columns=columns)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        columns = ['image_name', 'ground_truth', 'prediction']
        self.eval_table = self.wandb.Table(columns=columns)

    def _add_ground_truth(self, runner):
        # Get image loading pipeline
        from mmseg.datasets.pipelines import LoadImageFromFile
        img_loader = None
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, LoadImageFromFile):
                img_loader = t

        if img_loader is None:
            self.log_evaluation = False
            runner.logger.warning(
                'LoadImageFromFile is required to add images '
                'to W&B Tables.')
            return

        # Select the images to be logged.
        self.eval_image_indexs = np.arange(len(self.val_dataset))
        # Set seed so that same validation set is logged each time.
        np.random.seed(42)
        np.random.shuffle(self.eval_image_indexs)
        self.eval_image_indexs = self.eval_image_indexs[:self.num_eval_images]

        classes = self.val_dataset.CLASSES
        self.class_id_to_label = {id: name for id, name in enumerate(classes)}
        self.class_set = self.wandb.Classes([{
            'id': id,
            'name': name
        } for id, name in self.class_id_to_label.items()])

        for idx in self.eval_image_indexs:
            img_info = self.val_dataset.img_infos[idx]
            image_name = img_info['filename']

            # Get image and convert from BGR to RGB
            img_meta = img_loader(
                dict(img_info=img_info, img_prefix=self.val_dataset.img_dir))
            image = mmcv.bgr2rgb(img_meta['img'])

            # Get segmentation mask
            seg_mask = self.val_dataset.get_gt_seg_map_by_idx(idx)
            # Dict of masks to be logged.
            wandb_masks = None
            if seg_mask.ndim == 2:
                wandb_masks = {
                    'ground_truth': {
                        'mask_data': seg_mask,
                        'class_labels': self.class_id_to_label
                    }
                }

                # Log a row to the data table.
                self.data_table.add_data(
                    image_name,
                    self.wandb.Image(
                        image, masks=wandb_masks, classes=self.class_set))
            else:
                runner.logger.warning(
                    f'The segmentation mask is {seg_mask.ndim}D which '
                    'is not supported by W&B.')
                self.log_evaluation = False
                return

    def _log_predictions(self, results, runner):
        table_idxs = self.data_table_ref.get_index()
        assert len(table_idxs) == len(self.eval_image_indexs)
        assert len(results) == len(self.val_dataset)

        for ndx, eval_image_index in enumerate(self.eval_image_indexs):
            # Get the result
            pred_mask = results[eval_image_index]

            if pred_mask.ndim == 2:
                wandb_masks = {
                    'prediction': {
                        'mask_data': pred_mask,
                        'class_labels': self.class_id_to_label
                    }
                }

                # Log a row to the data table.
                self.eval_table.add_data(
                    self.data_table_ref.data[ndx][0],
                    self.data_table_ref.data[ndx][1],
                    self.wandb.Image(
                        self.data_table_ref.data[ndx][1],
                        masks=wandb_masks,
                        classes=self.class_set))
            else:
                runner.logger.warning(
                    'The predictio segmentation mask is '
                    f'{pred_mask.ndim}D which is not supported by W&B.')
                self.log_evaluation = False
                return

    def _log_data_table(self):
        """Log the W&B Tables for validation data as artifact and calls
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded images.

        This allows the data to be uploaded just once.
        """
        data_artifact = self.wandb.Artifact('val', type='dataset')
        data_artifact.add(self.data_table, 'val_data')

        self.wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        self.data_table_ref = data_artifact.get('val_data')

    def _log_eval_table(self, iter):
        """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        pred_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_pred', type='evaluation')
        pred_artifact.add(self.eval_table, 'eval_data')
        self.wandb.run.log_artifact(pred_artifact)
