# 将同一张图片的不同后缀的mask合并到同一张mask上
import glob
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

rp = './data/EndoCV2020_EDD/masks/train'

masks = glob.glob(os.path.join(rp, '*.png'))

original_mask_names = []
for imagep in tqdm(masks):
    image_name = os.path.basename(imagep).split('_')[0] + '_' \
        + os.path.basename(imagep).split('_')[1]
    original_mask_names.append(image_name)

original_mask_names = list(set(original_mask_names))

# # os.rmfile(os.path.join(rp, 'new_masks'))
# for omn in tqdm(original_mask_names):
#     if '.png' in omn:
#         os.remove(os.path.join(rp, omn))

for omn in tqdm(original_mask_names):
    # break
    masks = glob.glob(os.path.join(rp, omn + '_*.png'))
    # print(masks)
    # print('----------')
    # print(len(masks))
    # continue
    img = Image.open(masks[0])
    shape = np.array(img).shape
    new_mask = np.zeros([shape[0], shape[1]], dtype=np.uint8)
    # new_mask_2 = np.zeros([shape[0],shape[1]], dtype=np.uint8)
    # new_mask_3 = np.zeros([shape[0],shape[1]], dtype=np.uint8)
    for mask in masks:
        mask_name = os.path.basename(mask)
        if '_1.png' in mask_name:
            mask_1 = Image.open(mask).convert('L')
            mask_1 = np.array(mask_1)
            new_mask += mask_1 * 1
            # new_mask[:,:,0] = mask_1
        elif '_2.png' in mask_name:
            mask_2 = Image.open(mask).convert('L')
            mask_2 = np.array(mask_2)
            new_mask += mask_2 * 3
            # new_mask[:,:,1] = mask_2
        elif '_3.png' in mask_name:
            mask_3 = Image.open(mask).convert('L')
            mask_3 = np.array(mask_3)
            new_mask += mask_3 * 5
            # new_mask[:,:,2] = mask_3
        elif '_4.png' in mask_name:
            mask_4 = Image.open(mask).convert('L')
            mask_4 = np.array(mask_4)
            new_mask += mask_4 * 7
            # new_mask[:,:,3] = mask_4
        elif '_5.png' in mask_name:
            mask_5 = Image.open(mask).convert('L')
            mask_5 = np.array(mask_5)
            new_mask += mask_5 * 9
            # new_mask[:,:,4] = mask_5
        else:
            print('=======================================')
            continue

    # new_mask = np.dstack([mask_1, mask_2, mask_3, mask_4, mask_5])
    new_mask[(new_mask % 2 == 0) == (new_mask > 0)] = 255
    new_mask[new_mask == 3] = 2
    new_mask[new_mask == 5] = 3
    new_mask[new_mask == 7] = 4
    new_mask[new_mask == 9] = 5
    new_mask = Image.fromarray(new_mask.astype(np.uint8))
    new_mask.save(os.path.join(rp, omn + '.png'))
    # cv2.imwrite(os.path.join(rp, omn + '.png'), np.array(new_mask))
