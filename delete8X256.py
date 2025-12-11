import os
from PIL import Image

# 设置基础路径
base_folder = '../Floodnet/data/mixed_dataset/SAR/'  # 修改为你的SAR文件夹路径
subsets = ['train', 'val', 'test']

total_deleted = 0

for subset in subsets:
    image_folder = os.path.join(base_folder, subset, 'images')
    label_folder = os.path.join(base_folder, subset, 'labels')

    if not os.path.exists(image_folder):
        print(f'{image_folder} 不存在，跳过')
        continue

    deleted_count = 0

    for filename in os.listdir(image_folder):
        if filename.endswith('.tif'):
            img_path = os.path.join(image_folder, filename)

            # 读取图像尺寸
            with Image.open(img_path) as img:
                width, height = img.size

            # 如果不是256x256，删除图像和对应标签
            if width != 256 or height != 256:
                # 删除图像
                os.remove(img_path)

                # 删除对应标签
                label_filename = filename.replace('.tif', '.png')
                label_path = os.path.join(label_folder, label_filename)
                if os.path.exists(label_path):
                    os.remove(label_path)

                deleted_count += 1
                print(f'[{subset}] 已删除: {filename} (尺寸: {width}x{height})')

    print(f'{subset} 删除了 {deleted_count} 对文件\n')
    total_deleted += deleted_count

print(f'总共删除了 {total_deleted} 对文件')