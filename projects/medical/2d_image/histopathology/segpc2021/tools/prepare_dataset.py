import os
import shutil

from tqdm import tqdm

rp = './SegPC2021/SegPC2021/TCIA_SegPC_dataset/TCIA_SegPC_dataset/TCIA_SegPC_dataset/validation/validation/y'  # noqa

for im in tqdm(os.listdir(rp)):
    im_p = os.path.join(rp, im)
    shutil.copy(im_p, './SegPC2021/masks/val/' + im)

    im_im = im.split('_')[0] + '.bmp'
    im_im_p = os.path.join(
        './SegPC2021/SegPC2021/TCIA_SegPC_dataset/TCIA_SegPC_dataset/TCIA_SegPC_dataset/validation/validation/x',  # noqa
        im_im)
    shutil.copy(im_im_p, './SegPC2021/images/val/' + im)
