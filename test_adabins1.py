import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.cm
from mmseg.apis.inference import init_model


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480)):
        # image = image.resize(target_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


def colorize(value, vmin=10, vmax=1000, cmap='magma_r'):
    value = value.cpu().numpy()[0, :, :]
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    img = value[:, :, :3]

    return img


class InferenceHelper:
    def __init__(self, dataset='nyu', device='cuda:0'):
        self.toTensor = ToTensor()
        self.device = device
        if dataset == 'nyu':
            self.min_depth = 1e-3
            self.max_depth = 10
            self.saving_factor = 1000  # used to save in 16 bit
            pretrained_path = "./projects/Adabins/pretrained/AdaBins_nyu.pth"
            model = init_model('projects/Adabins/configs/adabins/adabins_efficient_b5_4x16_25e_NYU_416x544.py',
                               checkpoint=pretrained_path)

        elif dataset == 'kitti':
            self.min_depth = 1e-3
            self.max_depth = 80
            self.saving_factor = 256
            pretrained_path = "./projects/Adabins/pretrained/AdaBins_kitti.pth"
            model = init_model('projects/Adabins/configs/Adabins/adabins_efficient_b5_4x16_25e_kitti_352x704.py',
                               checkpoint=pretrained_path)
        else:
            raise ValueError("dataset can be either 'nyu' or 'kitti' but got {}".format(dataset))
        # model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict_pil(self, pil_image, visualized=False):
        # pil_image = pil_image.resize((640, 480))
        img = np.asarray(pil_image) / 255.

        img = self.toTensor(img).unsqueeze(0).float().to(self.device)
        bin_centers, pred = self.predict(img)

        if visualized:
            viz = colorize(torch.from_numpy(pred).unsqueeze(0), vmin=None, vmax=None, cmap='magma')
            # pred = np.asarray(pred*1000, dtype='uint16')
            viz = Image.fromarray(viz)
            return bin_centers, pred, viz
        return bin_centers, pred

    @torch.no_grad()
    def predict(self, image):
        print(image)
        bins, pred = self.model(image)
        print(pred)
        # from reprod_log import ReprodLogger
        # reprod_logger = ReprodLogger()
        # reprod_logger.add("logits", pred.cpu().detach().numpy())
        # reprod_logger.save("C:\\Users\\Administrator\\Desktop\\dgw\\forward_paddle.npy")
        print(pred.shape)
        # pred = np.clip(pred.cpu().numpy(), self.min_depth, self.max_depth)
        # # Flip
        image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(self.device)
        pred_lr = self.model(image)[-1]
        # pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], self.min_depth, self.max_depth)

        pred = torch.clamp(pred, self.min_depth, self.max_depth)
        pred_lr = torch.clamp(torch.flip(pred_lr, [-1]), self.min_depth, self.max_depth)
        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)
        final = nn.functional.interpolate(final, image.shape[-2:],
                                          mode='bilinear', align_corners=True)

        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[torch.isinf(final)] = self.max_depth
        final[torch.isnan(final)] = self.min_depth

        centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        centers = centers.cpu().squeeze().numpy()
        centers = centers[centers > self.min_depth]
        centers = centers[centers < self.max_depth]
        return centers, final

    @torch.no_grad()
    def predict_dir(self, test_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        transform = ToTensor()
        all_files = glob.glob(os.path.join(test_dir, "*"))
        self.model.eval()
        for f in tqdm(all_files):
            image = np.asarray(Image.open(f), dtype='float32') / 255.
            image = transform(image).unsqueeze(0).to(self.device)

            centers, final = self.predict(image)
            # final = final.squeeze().cpu().numpy()

            final = (final * self.saving_factor).astype('uint16')
            basename = os.path.basename(f).split('.')[0]
            save_path = os.path.join(out_dir, basename + ".png")

            Image.fromarray(final.squeeze()).save(save_path)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from time import time

    img = Image.open("demo/classroom__rgb_00283.jpg")
    start = time()
    inferHelper = InferenceHelper()
    centers, pred = inferHelper.predict_pil(img)
    print(f"took :{time() - start}s")
    plt.imshow(pred.cpu().numpy().squeeze(), cmap='magma_r')
    plt.show()
