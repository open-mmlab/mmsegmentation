import os
import shutil

from PIL import Image

path = 'data'

if not os.path.exists(os.path.join(path, 'images', 'train')):
    os.system(f'mkdir -p {os.path.join(path, "images", "train")}')

if not os.path.exists(os.path.join(path, 'masks', 'train')):
    os.system(f'mkdir -p {os.path.join(path, "masks", "train")}')

origin_data_path = os.path.join(path, 'vesselSegmentation')

imgs_amd14 = os.listdir(os.path.join(origin_data_path, 'AMD14'))
imgs_ger7 = os.listdir(os.path.join(origin_data_path, 'GER7'))

for img in imgs_amd14:
    shutil.copy(
        os.path.join(origin_data_path, 'AMD14', img),
        os.path.join(path, 'images', 'train', img))
    # copy GT
    img_gt = img.replace('.png', '-GT.png')
    shutil.copy(
        os.path.join(origin_data_path, 'AMD14-GT', f'{img_gt}'),
        os.path.join(path, 'masks', 'train', img))

for img in imgs_ger7:
    shutil.copy(
        os.path.join(origin_data_path, 'GER7', img),
        os.path.join(path, 'images', 'train', img))
    # copy GT
    img_gt = img.replace('.bmp', '-GT.png')
    img = img.replace('bmp', 'png')
    shutil.copy(
        os.path.join(origin_data_path, 'GER7-GT', img_gt),
        os.path.join(path, 'masks', 'train', img))

imgs = os.listdir(os.path.join(path, 'images', 'train'))
for img in imgs:
    if not img.endswith('.png'):
        im = Image.open(os.path.join(path, 'images', 'train', img))
        im.save(os.path.join(path, 'images', 'train', img[:-4] + '.png'))
