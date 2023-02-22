import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm


# map = {255:0, 128:1, 0:2}

imgs = glob.glob(os.path.join('./masks/train', '*.bmp'))

for im_path in tqdm(imgs):
    im = Image.open(im_path)
    imn = np.array(im)
    imn[imn == 255] = 0
    imn[imn == 128] = 1
    imn[imn == 0] = 2
    new_im = Image.fromarray(imn)
    new_im.save(im_path)


