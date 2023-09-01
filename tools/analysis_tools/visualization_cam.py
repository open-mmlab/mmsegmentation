"""
Use the pytorch-grad-cam tool to visualize Class Activation Maps (CAM).

requirement: pip install grad-cam
"""

import torch

import torch.nn.functional as F

import numpy as np

from PIL import Image

from argparse import ArgumentParser

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from pytorch_grad_cam import GradCAM

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot

from mmseg.utils import register_all_modules


class SemanticSegmentationTarget:
    """wrap the model.

    requirement: pip install grad-cam

    Args:
        category (int): Visualization class.
        mask (ndarray): Mask of class.
        size (tuple): Image size.
    """

    def __init__(self, category, mask, size):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.size = size
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        model_output = torch.unsqueeze(model_output, dim=0)
        model_output = F.interpolate(model_output,
                                     size=self.size,
                                     mode='bilinear')
        model_output = torch.squeeze(model_output, dim=0)

        return (model_output[self.category, :, :] * self.mask).sum()


def main():
    parser = ArgumentParser()

    parser.add_argument('--img', default='car.jpg',
                        help='Image file')
    parser.add_argument('--config',
                        default='configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-769x769.py',
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default='deeplabv3_r50-d8_769x769_40k_cityscapes.pth',
                        help='Checkpoint file')
    parser.add_argument('--out-file', default='prediction.png',
                        help='Path to output prediction file')
    parser.add_argument('--cam-file', default='vis_cam.png',
                        help='Path to output cam file')
    parser.add_argument('--device', default='cuda:0',
                        help='cuda:0 or cpu')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    register_all_modules()
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # test a single image
    result = inference_model(model, args.img)

    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)

    # result data conversion
    prediction_data = result.pred_sem_seg.data
    pre_np_data = prediction_data.cpu().numpy().squeeze(0)

    # select visualization layer, e.g. model.backbone.layer4[2] in deeplabv3_r50
    # it can be multiple layers
    target_layers = [model.backbone.layer4[2]]

    # select visualization class, e.g. traffic sign(index:7)
    car_category = 7  # traffic sign
    car_mask_float = np.float32(pre_np_data == car_category)

    # data processing
    image = np.array(Image.open(args.img))
    Height, Width = image.shape[0], image.shape[1]
    rgb_img = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # Grad CAM(Class Activation Maps)
    targets = [SemanticSegmentationTarget(car_category,
                                          car_mask_float,
                                          (Height, Width))]
    with GradCAM(model=model,
                 target_layers=target_layers,
                 use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img,
                                      grayscale_cam,
                                      use_rgb=True)

        # save cam file
        Image.fromarray(cam_image).save(args.cam_file)


if __name__ == '__main__':
    main()

