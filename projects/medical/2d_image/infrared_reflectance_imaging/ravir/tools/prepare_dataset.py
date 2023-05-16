import glob
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

# map = {255:2, 128:1, 0:0}

os.makedirs('data/ravir/images/train', exist_ok=True)
os.makedirs('data/ravir/images/test', exist_ok=True)
os.makedirs('data/ravir/masks/train', exist_ok=True)

os.system(
    r'cp data/ravir/RAVIR\ Dataset/train/training_images/* data/ravir/images/train'  # noqa
)
os.system(
    r'cp data/ravir/RAVIR\ Dataset/train/training_masks/* data/ravir/masks/train'  # noqa
)
os.system(r'cp data/ravir/RAVIR\ Dataset/test/* data/ravir/images/test')

os.system(r'rm -rf data/ravir/RAVIR\ Dataset')

imgs = glob.glob(os.path.join('data/ravir/masks/train', '*.png'))

for im_path in tqdm(imgs):
    im = Image.open(im_path)
    imn = np.array(im)
    imn[imn == 255] = 2
    imn[imn == 128] = 1
    imn[imn == 0] = 0
    new_im = Image.fromarray(imn)
    new_im.save(im_path)
