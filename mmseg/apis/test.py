import os.path as osp

import mmcv
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info


def single_gpu_test(model, data_loader, show=False, out_dir=None, opacity=0.5):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        list: evaluation preparetion results.
    """
    model.eval()
    pre_eval_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        pre_eval_results.extend(dataset.pre_eval(result, batch_indices))

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return pre_eval_results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: evaluation preparetion results.
    """
    model.eval()
    pre_eval_results = []
    dataset = data_loader.dataset

    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # TODO: adapt samples_per_gpu > 1.
        # only samples_per_gpu=1 valid now
        pre_eval_results.extend(dataset.pre_eval(result, batch_indices))

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        pre_eval_results = collect_results_gpu(pre_eval_results, len(dataset))
    else:
        pre_eval_results = collect_results_cpu(pre_eval_results, len(dataset),
                                               tmpdir)
    return pre_eval_results
