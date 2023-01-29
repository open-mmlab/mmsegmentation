# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import MMSegInferencer


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('model', help='Config file')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-file', default='', help='Path to save result file')
    parser.add_argument(
        '--img-out-dir', default='', help='Path to save painted img')
    parser.add_argument(
        '--save-mask',
        action='store_true',
        default=False,
        help='Enable save the mask file')
    parser.add_argument(
        '--palette',
        default=None,
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    mmseg_inferencer = MMSegInferencer(
        args.model, args.checkpoint, palette=args.palette, device='cuda:0')

    # test a single image
    mmseg_inferencer(
        args.img,
        show=args.save_mask,
        img_out_dir=args.img_out_dir,
        save_mask=args.save_mask,
        pred_out_file=args.out_file,
        opacity=args.opacity)


if __name__ == '__main__':
    main()
