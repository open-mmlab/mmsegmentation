import os

masks = os.listdir(
    '/mnt/lustre/guosizheng/gsz/gsz_copy/to_ceph/EAD19/masks/train')

imgs = os.listdir(
    '/mnt/lustre/guosizheng/gsz/gsz_copy/to_ceph/EAD19/images/train')

for img in imgs:
    if img not in masks:
        print(img)
