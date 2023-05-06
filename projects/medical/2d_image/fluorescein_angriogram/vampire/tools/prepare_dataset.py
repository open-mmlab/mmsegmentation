import os
import shutil

from PIL import Image

path = 'path/to/vampire'

if not os.path.exists(os.path.join(path, 'images', 'train')):
    os.mkdirs(os.path.join(path, 'images', 'train'))

if not os.path.exists(os.path.join(path, 'masks', 'train')):
    os.mkdirs(os.path.join(path, 'masks', 'train'))

origin_data_path = os.path.join(path, 'vesselSegmentation')

imgs_amd14 = os.listdir(os.path.join(origin_data_path, 'AMD14'))
imgs_ger7 = os.listdir(os.path.join(origin_data_path, 'GER7'))

for img in imgs_amd14:
    shutil.copy(
        os.path.join(origin_data_path, 'AMD14', img),
        os.path.join(path, 'images', 'train', img))
    shutil.copy(
        os.path.join(origin_data_path, 'AMD14-GT', img),
        os.path.join(path, 'masks', 'train', img))

for img in imgs_ger7:
    shutil.copy(
        os.path.join(origin_data_path, 'GER7', img),
        os.path.join(path, 'images', 'train', img))
    shutil.copy(
        os.path.join(origin_data_path, 'GER7-GT', img),
        os.path.join(path, 'masks', 'train', img))

imgs = os.listdir(os.path.join(path, 'images', 'train'))
for img in imgs:
    if not img.endswith('.png'):
        im = Image.open(os.path.join(origin_data_path, 'images', 'train', img))
        im.save(os.path.join(path, 'images', 'train', img[:-4] + '.png'))
