import argparse
import os
import warnings

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info
from mmcv.utils import DictAction

from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models.segmentors.base import BaseSegmentor
from mmcv.tensorrt import TRTWraper, load_tensorrt_plugin
from typing import Any, Iterable


class TensorRTSegmentor(BaseSegmentor):

    def __init__(self, trt_file: str, cfg: Any, device_id: int):
        super(TensorRTSegmentor, self).__init__()
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with TensorRT from source.')
        model = TRTWraper(
            trt_file, input_names=['input'], output_names=['output'])

        self.model = model
        self.device_id = device_id
        self.cfg = cfg
        self.test_mode = cfg.model.test_cfg.mode

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def encode_decode(self, img, img_metas):
        raise NotImplementedError('This method is not implemented.')

    def forward_train(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def simple_test(self, img: torch.Tensor, img_meta: Iterable,
                    **kwargs) -> list:
        with torch.cuda.device(self.device_id), torch.no_grad():
            seg_pred = self.model({'input': img})['output']
        seg_pred = seg_pred.detach().cpu().numpy()
        # whole might support dynamic reshape
        ori_shape = img_meta[0]['ori_shape']
        if not (ori_shape[0] == seg_pred.shape[-2]
                and ori_shape[1] == seg_pred.shape[-1]):
            seg_pred = torch.from_numpy(seg_pred).float()
            seg_pred = torch.nn.functional.interpolate(
                seg_pred, size=tuple(ori_shape[:2]), mode='nearest')
            seg_pred = seg_pred.long().detach().cpu().numpy()
        seg_pred = seg_pred[0]
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='mmseg tensorrt backend test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('model', help='Input model file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # load onnx config and meta
    cfg.model.train_cfg = None
    model = TensorRTSegmentor(args.model, cfg=cfg, device_id=0)
    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                              efficient_test, args.opacity)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()
