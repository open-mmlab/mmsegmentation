import glob
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def map_func(num_map={255: 1, 0: 0}, im_path='./masks/train', suffix='png'):

    imgs = glob.glob(os.path.join(im_path, f'*.{suffix}'))

    for im_path in tqdm(imgs):
        im = Image.open(im_path)
        imn = np.array(im)
        for num in num_map:
            imn[imn == num] = num_map[num]
        new_im = Image.fromarray(imn)
        new_im.save(im_path)


if __name__ == '__main__':
    num_map = {255: 1, 0: 0}
    suffix = 'png'
    map_func(num_map, 'path/to/data/masks/train', suffix)
    map_func(num_map, 'path/to/data/masks/test', suffix)
