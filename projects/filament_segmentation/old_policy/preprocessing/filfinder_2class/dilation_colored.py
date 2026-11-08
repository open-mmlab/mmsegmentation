import cv2
import numpy as np



# 読み込んだスケルトン画像（背景が黒、検出部分が白の形式）
fil_skeleton = cv2.imread(fits_name + "_fil_skeleton.png", cv2.IMREAD_GRAYSCALE)

# 背景を白、検出部分を赤に変換
h, w = fil_skeleton.shape
colored_img = np.ones((h, w, 3), dtype=np.uint8) * 255  # 背景を白 (255, 255, 255)
colored_img[fil_skeleton == 255] = [255, 0, 0]  # 検出部分を赤 (255, 0, 0)

# 保存
cv2.imwrite(fits_name + "_fil_skeleton_colored.png", colored_img)

# 3×3 の膨張処理
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(fil_skeleton, kernel, iterations=1)

# 膨張後の画像を背景白、検出部分赤に再変換
dilated_colored_img = np.ones((h, w, 3), dtype=np.uint8) * 255  # 背景を白
dilated_colored_img[dilated == 255] = [255, 0, 0]  # 検出部分を赤

# 保存
cv2.imwrite(fits_name + "_fil_skeleton_dilated_colored.png", dilated_colored_img)
