import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
import random



# 设置路径
gt_folder = '/mnt/d//Data/DLdata/urban_sar_floods/urban_sar_floods/03_FU/GT'  # 标签文件夹
sar_folder = '/mnt/d//Data/DLdata/urban_sar_floods/urban_sar_floods/03_FU/SAR'  # 影像文件夹
output_folder = '/mnt/d//Data/DLdata/urban_sar_floods/urban_sar_floods/03_FU/output'  # 输出文件夹

# 创建输出目录结构
for subset in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_folder, subset, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, subset, 'labels'), exist_ok=True)


# 裁剪函数 - 适配多波段
def crop_image(img_array, crop_size=256):
    crops = []
    if len(img_array.shape) == 3:  # 多波段 (bands, height, width)
        bands, h, w = img_array.shape
        for i in range(0, h, crop_size):
            for j in range(0, w, crop_size):
                crop = img_array[:, i:i + crop_size, j:j + crop_size]
                if crop.shape[1] == crop_size and crop.shape[2] == crop_size:
                    crops.append(crop)
    else:  # 单波段 (height, width)
        h, w = img_array.shape
        for i in range(0, h, crop_size):
            for j in range(0, w, crop_size):
                crop = img_array[i:i + crop_size, j:j + crop_size]
                if crop.shape[0] == crop_size and crop.shape[1] == crop_size:
                    crops.append(crop)
    return crops


# 获取所有文件
gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.tif')])
sar_files = sorted([f for f in os.listdir(sar_folder) if f.endswith('.tif')])

# 生成所有裁剪块
all_crops = []
for gt_file, sar_file in zip(gt_files, sar_files):
    gt_path = os.path.join(gt_folder, gt_file)
    sar_path = os.path.join(sar_folder, sar_file)

    # 读取标签（单波段）
    with Image.open(gt_path) as gt_img:
        gt_array = np.array(gt_img)

    # 读取SAR影像（多波段）
    with rasterio.open(sar_path) as sar_src:
        sar_array = sar_src.read()  # 读取所有波段 (bands, height, width)
        sar_profile = sar_src.profile

    # 裁剪
    gt_crops = crop_image(gt_array)
    sar_crops = crop_image(sar_array)

    # 保存裁剪信息
    base_name = os.path.splitext(gt_file)[0]
    for idx, (gt_crop, sar_crop) in enumerate(zip(gt_crops, sar_crops)):
        all_crops.append({
            'gt': gt_crop,
            'sar': sar_crop,
            'name': f'{base_name}_{idx}',
            'profile': sar_profile
        })

print(f'共生成 {len(all_crops)} 个裁剪块')

# 打乱数据
random.shuffle(all_crops)

# 按6:2:2划分
total = len(all_crops)
train_end = int(total * 0.6)
val_end = int(total * 0.8)

train_crops = all_crops[:train_end]
val_crops = all_crops[train_end:val_end]
test_crops = all_crops[val_end:]


# 保存函数
def save_crops(crops, subset):
    for crop_data in crops:
        # 保存多波段SAR影像为TIF
        sar_path = os.path.join(output_folder, subset, 'images', f"sar_{crop_data['name']}.tif")

        # 更新profile
        profile = crop_data['profile'].copy()
        profile.update({
            'height': 256,
            'width': 256,
            'transform': from_bounds(0, 0, 256, 256, 256, 256)
        })

        with rasterio.open(sar_path, 'w', **profile) as dst:
            dst.write(crop_data['sar'])

        # 处理标签：将值为2的像元改为1
        gt_array = crop_data['gt'].copy()
        gt_array[gt_array == 2] = 1

        # 保存标签为PNG
        gt_img = Image.fromarray(gt_array.astype(np.uint8))
        gt_path = os.path.join(output_folder, subset, 'labels', f"sar_{crop_data['name']}.png")
        gt_img.save(gt_path)

    print(f'{subset}: {len(crops)} 张图像')


# 保存所有数据集
save_crops(train_crops, 'train')
save_crops(val_crops, 'val')
save_crops(test_crops, 'test')

print(f'\n总共处理了 {total} 张裁剪图像')
print(f'train: {len(train_crops)}, val: {len(val_crops)}, test: {len(test_crops)}')