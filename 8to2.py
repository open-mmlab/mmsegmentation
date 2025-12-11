import os
import numpy as np
import rasterio
from shutil import copy2
from tqdm import tqdm


def keep_last_two_bands(folder_path, backup=True, backup_suffix='_8band_backup'):
    """
    将所有TIF文件只保留最后两个波段，覆盖原文件

    参数:
        folder_path: 要处理的根目录路径
        backup: 是否备份原文件
        backup_suffix: 备份文件后缀
    """
    total_files = 0
    files_processed = 0
    files_skipped = 0
    tif_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']

    # 收集所有tif文件
    print(f"扫描路径: {folder_path}")
    tif_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in tif_extensions):
                tif_files.append(os.path.join(root, file))

    total_files = len(tif_files)
    print(f"找到 {total_files} 个TIF文件\n")

    # 第一步：检查文件波段数
    print("步骤1: 检查文件波段数...")
    print("=" * 80)

    band_info = {}
    files_to_process = []

    try:
        iterator = tqdm(tif_files, desc="扫描文件")
    except:
        iterator = tif_files

    for file_path in iterator:
        try:
            with rasterio.open(file_path) as src:
                num_bands = src.count
                rel_path = os.path.relpath(file_path, folder_path)

                if num_bands not in band_info:
                    band_info[num_bands] = []
                band_info[num_bands].append(rel_path)

                if num_bands >= 2:  # 只处理波段数>=2的文件
                    files_to_process.append((file_path, num_bands))
                else:
                    files_skipped += 1

        except Exception as e:
            print(f"\n读取错误: {os.path.basename(file_path)}")
            print(f"  错误: {e}")

    # 显示波段数统计
    print("\n波段数统计:")
    for num_bands in sorted(band_info.keys()):
        print(f"  {num_bands}波段: {len(band_info[num_bands])} 个文件")

    print(f"\n将处理 {len(files_to_process)} 个文件（波段数>=2）")
    print(f"跳过 {files_skipped} 个文件（波段数<2）")

    if len(files_to_process) == 0:
        print("没有需要处理的文件！")
        return

    # 第二步：确认并处理
    print("\n" + "=" * 80)
    print("步骤2: 处理文件")
    print("操作: 只保留最后两个波段，覆盖原文件")

    if backup:
        print(f"将创建备份文件（后缀: {backup_suffix}）")
    else:
        print("警告: 不会创建备份！")

    user_input = input("\n是否继续？(y/n): ")
    if user_input.lower() != 'y':
        print("操作已取消")
        return

    print("\n开始处理...")

    try:
        iterator = tqdm(files_to_process, desc="处理文件")
    except:
        iterator = files_to_process

    for file_path, original_bands in iterator:
        try:
            # 备份原文件
            if backup:
                backup_path = file_path + backup_suffix
                copy2(file_path, backup_path)

            # 读取文件
            with rasterio.open(file_path) as src:
                # 保存元数据并修改波段数
                meta = src.meta.copy()
                meta['count'] = 2  # 修改为2波段

                # 读取最后两个波段
                band_n_1 = src.read(original_bands - 1)  # 倒数第二个波段
                band_n = src.read(original_bands)  # 最后一个波段

                # 创建临时文件
                temp_path = file_path + '.tmp'

                # 写入新文件（只有2个波段）
                with rasterio.open(temp_path, 'w', **meta) as dst:
                    dst.write(band_n_1, 1)  # 写入第1个波段
                    dst.write(band_n, 2)  # 写入第2个波段

            # 替换原文件
            os.replace(temp_path, file_path)
            files_processed += 1

        except Exception as e:
            print(f"\n处理错误: {os.path.basename(file_path)}")
            print(f"  原波段数: {original_bands}")
            print(f"  错误: {e}")
            # 清理临时文件
            temp_path = file_path + '.tmp'
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 输出结果
    print("\n" + "=" * 80)
    print("处理完成!")
    print(f"总文件数: {total_files}")
    print(f"成功处理的文件: {files_processed}")
    print(f"跳过的文件: {files_skipped}")

    if backup:
        print(f"\n原始文件已备份（后缀: {backup_suffix}）")
        print("如果确认无误，可以使用以下命令删除备份文件:")
        print(f"  find {folder_path} -name '*{backup_suffix}' -delete")

    # 验证处理结果
    print("\n步骤3: 验证处理结果...")
    print("检查前3个文件的波段数...")

    for i, (file_path, _) in enumerate(files_to_process[:3]):
        try:
            with rasterio.open(file_path) as src:
                rel_path = os.path.relpath(file_path, folder_path)
                print(f"  ✓ {rel_path}: {src.count} 波段")
        except Exception as e:
            print(f"  ✗ 验证失败: {e}")

        if i >= 2:  # 只检查前3个
            break


if __name__ == "__main__":
    folder_path = "/mnt/d/Project/Code/Floodnet/data/mixed_dataset/SAR/"

    # 执行处理
    # backup=True 会备份原文件（推荐）
    # backup=False 直接覆盖（不推荐）
    keep_last_two_bands(folder_path, backup=False, backup_suffix='_8band_backup')