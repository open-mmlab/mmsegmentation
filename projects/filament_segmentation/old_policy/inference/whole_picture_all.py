"""
推論 → パッチつなぎ合わせ → 可視化 統合スクリプト

処理の流れ:
  1. orionBのパッチ画像を推論してパッチ単位の予測マスクを保存
  2. パッチ予測マスクを全面画像につなぎ合わせ
  3. 元画像・真値マスク・推論結果を重ね合わせた確認用画像を保存
"""

from mmseg.apis import inference_model, init_model
import numpy as np
import cv2
import os
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# -----------------------------------------------------------------
# 設定
# -----------------------------------------------------------------
CONFIG_FILE     = 'configs/unet/unet_deeplabv3_my_dataset.py'
CHECKPOINT_FILE = 'work_dirs/unet_deeplabv3_hgbs/iter_40000.pth'

ROOT_DIR  = '/mnt/sp4t/fujiwara.r'
FITS_NAME = 'HGBS_orionB_hires_column_density_map'

IN_DIR       = os.path.join(ROOT_DIR, FITS_NAME, 'in')
GT_MASK_PATH = os.path.join(ROOT_DIR, FITS_NAME, 'full', FITS_NAME + '_label_255.png')
IN_IMG_PATH  = os.path.join(ROOT_DIR, FITS_NAME, 'full', FITS_NAME + '_input_full_ml.png')

PATCH_OUT_DIR   = os.path.join(ROOT_DIR, FITS_NAME, 'pred_patches')
MERGED_OUT_PATH = os.path.join(ROOT_DIR, FITS_NAME, 'vis', FITS_NAME + '_pred_merged.png')
BLEND_OUT_PATH  = os.path.join(ROOT_DIR, FITS_NAME, 'vis', FITS_NAME + '_blend_result.png')

TILE_SIZE = 256
STEP_RATE = 0.0625

os.makedirs(PATCH_OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, FITS_NAME, 'vis'), exist_ok=True)


# -----------------------------------------------------------------
# Step 1: パッチ推論
# -----------------------------------------------------------------
def run_inference():
    print("=== Step 1: 推論 ===")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')

    img_paths = []
    for root, dirs, files in os.walk(IN_DIR):
        for fname in sorted(files):
            if fname.endswith('_in.png'):
                img_paths.append(os.path.join(root, fname))

    print(f"Total patches: {len(img_paths)}")

    for i, img_path in enumerate(img_paths):
        result   = inference_model(model, img_path)
        pred     = result.pred_sem_seg.data.cpu().numpy()[0]
        pred_img = (pred * 255).astype(np.uint8)
        fname     = os.path.basename(img_path)
        save_path = os.path.join(PATCH_OUT_DIR, fname.replace('_in.png', '_pred.png'))
        cv2.imwrite(save_path, pred_img)
        if i % 500 == 0:
            print(f"  {i}/{len(img_paths)}")

    print(f"推論完了: {len(img_paths)} patches -> {PATCH_OUT_DIR}")
    return img_paths


# -----------------------------------------------------------------
# Step 2: パッチをつなぎ合わせて全面画像を作成
# -----------------------------------------------------------------
def merge_patches(img_paths):
    print("=== Step 2: つなぎ合わせ ===")

    gt_mask = cv2.imread(GT_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise FileNotFoundError(f"真値マスクが見つかりません: {GT_MASK_PATH}")
    H, W = gt_mask.shape
    print(f"全面画像サイズ: {W}x{H}")

    merged    = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    pattern = re.compile(r'_(\d{5})-(\d{5})_in\.png$')

    for img_path in img_paths:
        fname = os.path.basename(img_path)
        m = pattern.search(fname)
        if m is None:
            print(f"WARNING: 座標取得失敗: {fname}")
            continue
        x = int(m.group(1))
        y = int(m.group(2))

        pred_path = os.path.join(PATCH_OUT_DIR, fname.replace('_in.png', '_pred.png'))
        if not os.path.exists(pred_path):
            continue

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        y_end = min(y + TILE_SIZE, H)
        x_end = min(x + TILE_SIZE, W)
        ph = y_end - y
        pw = x_end - x

        merged[y:y_end, x:x_end]    += pred[:ph, :pw]
        count_map[y:y_end, x:x_end] += 1.0

    valid = count_map > 0
    merged[valid] /= count_map[valid]

    merged_img = (merged * 255).astype(np.uint8)
    cv2.imwrite(MERGED_OUT_PATH, merged_img)
    print(f"つなぎ合わせ完了: {MERGED_OUT_PATH}")
    return merged_img


# -----------------------------------------------------------------
# Step 3: 元画像・真値・推論結果を重ね合わせて可視化
# -----------------------------------------------------------------
def blend_result():
    print("=== Step 3: 可視化 ===")

    img = cv2.imread(IN_IMG_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"元画像が見つかりません: {IN_IMG_PATH}")

    gt_mask = cv2.imread(GT_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise FileNotFoundError(f"真値マスクが見つかりません: {GT_MASK_PATH}")

    pred_mask = cv2.imread(MERGED_OUT_PATH, cv2.IMREAD_GRAYSCALE)
    if pred_mask is None:
        raise FileNotFoundError(f"推論結果が見つかりません: {MERGED_OUT_PATH}")

    H = min(img.shape[0], gt_mask.shape[0], pred_mask.shape[0])
    W = min(img.shape[1], gt_mask.shape[1], pred_mask.shape[1])
    img       = img[:H, :W]
    gt_mask   = gt_mask[:H, :W]
    pred_mask = pred_mask[:H, :W]

    # 膨張処理で視認性を上げる
    kernel = np.ones((3, 3), np.uint8)
    gt_bin_vis   = cv2.dilate((gt_mask   > 0).astype(np.uint8), kernel, iterations=1).astype(bool)
    pred_bin_vis = cv2.dilate((pred_mask > 127).astype(np.uint8), kernel, iterations=1).astype(bool)

    # 3領域に分類
    gt_only   = gt_bin_vis & ~pred_bin_vis  # 赤:   ラベルのみ
    pred_only = pred_bin_vis & ~gt_bin_vis  # シアン: 推論のみ
    both      = gt_bin_vis & pred_bin_vis   # 白:   一致部分

    # カラーマスク作成（BGR）
    color_mask = np.zeros_like(img)
    color_mask[gt_only]   = (0,   0,   255)  # 赤
    color_mask[pred_only] = (255, 255, 0)    # シアン
    color_mask[both]      = (255, 255, 255)  # 白

    blend = cv2.addWeighted(img, 0.6, color_mask, 0.8, 0.0)

    ret = cv2.imwrite(BLEND_OUT_PATH, blend)
    if ret:
        print(f"可視化完了: {BLEND_OUT_PATH}")
    else:
        print(f"ERROR: 保存失敗: {BLEND_OUT_PATH}")

    # -----------------------------------------------------------------
    # 評価指標の計算（filamentクラスのみ、膨張なし）
    # -----------------------------------------------------------------
    gt_bin   = (gt_mask > 0)
    pred_bin_bool = (pred_mask > 127)

    TP = np.logical_and(gt_bin, pred_bin_bool).sum()
    FP = np.logical_and(~gt_bin, pred_bin_bool).sum()
    FN = np.logical_and(gt_bin, ~pred_bin_bool).sum()
    TN = np.logical_and(~gt_bin, ~pred_bin_bool).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("=== 評価結果（filamentクラス） ===")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  IoU       : {iou:.4f}")
    print(f"  F1 (Dice) : {f1:.4f}")
    print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")

    # CSVに保存
    import csv
    csv_path = os.path.join(ROOT_DIR, FITS_NAME, 'vis', FITS_NAME + '_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Precision', 'Recall', 'IoU', 'F1', 'TP', 'FP', 'FN', 'TN'])
        writer.writerow([f'{precision:.4f}', f'{recall:.4f}', f'{iou:.4f}', f'{f1:.4f}',
                         TP, FP, FN, TN])
    print(f"メトリクス保存: {csv_path}")


# -----------------------------------------------------------------
# メイン
# -----------------------------------------------------------------
if __name__ == '__main__':
    print("PATCH_OUT_DIR :", PATCH_OUT_DIR)
    print("MERGED_OUT_PATH:", MERGED_OUT_PATH)
    print("BLEND_OUT_PATH :", BLEND_OUT_PATH)

    img_paths = run_inference()
    merge_patches(img_paths)
    blend_result()
    print("=== 完了 ===")
