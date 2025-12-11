import os
import numpy as np
import rasterio
from shutil import copy2
from tqdm import tqdm


def replace_nan_with_zero(folder_path, backup=False, backup_suffix='_backup'):
    """
    遍历文件夹，将所有TIF文件中的NaN值替换为0

    参数:
        folder_path: 要处理的根目录路径
        backup: 是否备份原文件
        backup_suffix: 备份文件后缀
    """
    total_files = 0
    files_with_nan = 0
    files_processed = 0
    tif_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']

    # 收集所有tif文件
    tif_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in tif_extensions):
                tif_files.append(os.path.join(root, file))

    total_files = len(tif_files)
    print(f"找到 {total_files} 个TIF文件")
    print("=" * 80)

    # 第一步：检查哪些文件包含NaN
    print("\n步骤1: 检查NaN值...")
    files_to_process = []

    try:
        iterator = tqdm(tif_files, desc="扫描文件")
    except:
        iterator = tif_files

    for file_path in iterator:
        try:
            with rasterio.open(file_path) as src:
                has_nan = False
                for band_idx in range(1, src.count + 1):
                    band_data = src.read(band_idx)
                    if np.isnan(band_data).any():
                        has_nan = True
                        break

                if has_nan:
                    files_with_nan += 1
                    files_to_process.append(file_path)

        except Exception as e:
            print(f"\n读取错误: {file_path}")
            print(f"  错误: {e}")

    print(f"\n检查完成: {files_with_nan} 个文件包含NaN值")

    if files_with_nan == 0:
        print("没有需要处理的文件！")
        return

    # 第二步：处理包含NaN的文件
    print(f"\n步骤2: 替换NaN值为0...")
    print(f"需要处理 {len(files_to_process)} 个文件")

    if backup:
        print(f"将创建备份文件（后缀: {backup_suffix}）")

    user_input = input("\n是否继续？(y/n): ")
    if user_input.lower() != 'y':
        print("操作已取消")
        return

    print("\n开始处理...")

    try:
        iterator = tqdm(files_to_process, desc="处理文件")
    except:
        iterator = files_to_process

    for file_path in iterator:
        try:
            # 备份原文件
            if backup:
                backup_path = file_path + backup_suffix
                copy2(file_path, backup_path)

            # 读取文件
            with rasterio.open(file_path) as src:
                # 保存元数据
                meta = src.meta.copy()
                num_bands = src.count

                # 创建临时文件路径
                temp_path = file_path + '.tmp'

                # 写入处理后的数据
                with rasterio.open(temp_path, 'w', **meta) as dst:
                    for band_idx in range(1, num_bands + 1):
                        # 读取波段数据
                        band_data = src.read(band_idx)

                        # 将NaN替换为0
                        band_data = np.nan_to_num(band_data, nan=0.0)

                        # 写入波段
                        dst.write(band_data, band_idx)

            # 替换原文件
            os.replace(temp_path, file_path)
            files_processed += 1

        except Exception as e:
            print(f"\n处理错误: {file_path}")
            print(f"  错误: {e}")
            # 如果临时文件存在，删除它
            temp_path = file_path + '.tmp'
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 输出结果
    print("\n" + "=" * 80)
    print("处理完成!")
    print(f"总文件数: {total_files}")
    print(f"包含NaN的文件: {files_with_nan}")
    print(f"成功处理的文件: {files_processed}")

    if backup:
        print(f"\n原始文件已备份（后缀: {backup_suffix}）")
        print("如果确认无误，可以删除备份文件")


if __name__ == "__main__":
    folder_path = "/mnt/d/Project/Code/Floodnet/data/mixed_dataset/val/"

    # 执行替换
    # backup=True 会备份原文件
    # backup=False 直接覆盖原文件（不推荐）
    replace_nan_with_zero(folder_path, backup=False, backup_suffix='_backup')