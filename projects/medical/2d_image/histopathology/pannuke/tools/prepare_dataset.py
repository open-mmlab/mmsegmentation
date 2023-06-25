import os

import numpy as np
from PIL import Image

root_path = 'data/'

tgt_img_dir = os.path.join(root_path, 'images/train')
tgt_mask_dir = os.path.join(root_path, 'masks/train')
os.system('mkdir -p ' + tgt_img_dir)
os.system('mkdir -p ' + tgt_mask_dir)

fold_img_paths = sorted([
    os.path.join(root_path, 'pannuke/Fold 1/images/fold1/images.npy'),
    os.path.join(root_path, 'pannuke/Fold 2/images/fold2/images.npy'),
    os.path.join(root_path, 'pannuke/Fold 3/images/fold3/images.npy')
])

fold_mask_paths = sorted([
    os.path.join(root_path, 'pannuke/Fold 1/masks/fold1/masks.npy'),
    os.path.join(root_path, 'pannuke/Fold 2/masks/fold2/masks.npy'),
    os.path.join(root_path, 'pannuke/Fold 3/masks/fold3/masks.npy')
])

for n, (img_path,
        mask_path) in enumerate(zip(fold_img_paths, fold_mask_paths)):
    fold_name = str(n + 1)
    imgs = np.load(img_path)
    masks = np.load(mask_path)

    for i in range(imgs.shape[0]):
        img = np.uint8(imgs[i])
        mask_multichannel = np.minimum(np.uint8(masks[i]), 1)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for j in range(mask_multichannel.shape[-1]):
            factor = (j + 1) % mask_multichannel.shape[-1]
            # convert [0,1,2,3,4,5] to [1,2,3,4,5,0],
            # with the last label being background
            mask[mask_multichannel[..., j] == 1] = factor

        file_name = 'fold' + fold_name + '_' + str(i).rjust(4, '0') + '.png'
        print('Processing: ', file_name)
        tgt_img_path = os.path.join(tgt_img_dir, file_name)
        tgt_mask_path = os.path.join(tgt_mask_dir, file_name)
        Image.fromarray(img).save(tgt_img_path)
        Image.fromarray(mask).save(tgt_mask_path)

    del imgs
    del masks
