"""
推論 → パッチつなぎ合わせ → 可視化 統合スクリプト

使い方:
  # orionB（test）
  python infer_and_blend.py orionB

  # aquila（train）
  python infer_and_blend.py aquila

  # ic5146（train）
  python infer_and_blend.py ic5146

  # 全リージョン
  python infer_and_blend.py all

可視化の色分け:
  赤:   FilFinderラベルのみ（見逃し）
  シアン: 推論結果のみ（過検出）
  白:   ラベルと推論結果の一致部分
"""

from mmseg.apis import inference_model, init_model
import numpy as np
import cv2
import os
import re
import csv
import sys
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# -----------------------------------------------------------------
# 設定
# -----------------------------------------------------------------
CONFIG_FILE     = 'configs/unet/unet_deeplabv3_my_dataset.py'
CHECKPOINT_FILE = 'work_dirs/unet_deeplabv3_hgbs/iter_40000.pth'
ROOT_DIR        = '/mnt/sp4t/fujiwara.r'
TILE_SIZE       = 256
STEP_RATE       = 0.0625

REGION_MAP = {
    'aquila' : 'HGBS_aquilaM2_hires_column_density_map',
    'ic5146' : 'HGBS_ic5146_hires_column_density_map',
    'orionB' : 'HGBS_orionB_hires_column_density_map',
}


# -----------------------------------------------------------------
# ユーティリティ
# -----------------------------------------------------------------
def get_paths(fits_name):
    base = os.path.join(ROOT_DIR, fits_name)
    return dict(
        in_dir        = os.path.join(base, 'in'),
        gt_mask       = os.path.join(base, 'full', fits_name + '_label_255.png'),
        in_img        = os.path.join(base, 'full', fits_name + '_input_full_ml.png'),
        patch_out_dir = os.path.join(base, 'pred_patches'),
        merged        = os.path.join(base, 'vis', fits_name + '_pred_merged.png'),
        blend         = os.path.join(base, 'vis', fits_name + '_blend_result.png'),
        metrics_csv   = os.path.join(base, 'vis', fits_name + '_metrics.csv'),
        vis_dir       = os.path.join(base, 'vis'),
    )


# -----------------------------------------------------------------
# Step 1: パッチ推論
# -----------------------------------------------------------------
def run_inference(model, paths):
    print("=== Step 1: 推論 ===")
    # ←追加（前回結果削除）
    if os.path.exists(paths['patch_out_dir']):
        shutil.rmtree(paths['patch_out_dir'])
    os.makedirs(paths['patch_out_dir'], exist_ok=True)

    img_paths = []
    for root, dirs, files in os.walk(paths['in_dir']):
        for fname in sorted(files):
            if fname.endswith('_in.png'):
                img_paths.append(os.path.join(root, fname))

    print(f"Total patches: {len(img_paths)}")

    for i, img_path in enumerate(img_paths):
        result   = inference_model(model, img_path)
        pred     = result.pred_sem_seg.data.cpu().numpy()[0]
        pred_img = (pred * 255).astype(np.uint8)
        fname     = os.path.basename(img_path)
        save_path = os.path.join(paths['patch_out_dir'], fname.replace('_in.png', '_pred.png'))
        cv2.imwrite(save_path, pred_img)
        if i % 500 == 0:
            print(f"  {i}/{len(img_paths)}")

    print(f"推論完了: {len(img_paths)} patches -> {paths['patch_out_dir']}")
    return img_paths


# -----------------------------------------------------------------
# Step 2: パッチをつなぎ合わせて全面画像を作成
# -----------------------------------------------------------------
def merge_patches(img_paths, paths):
    print("=== Step 2: つなぎ合わせ ===")

    gt_mask = cv2.imread(paths['gt_mask'], cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise FileNotFoundError(f"真値マスクが見つかりません: {paths['gt_mask']}")
    H, W = gt_mask.shape
    print(f"全面画像サイズ: {W}x{H}")

    merged    = np.zeros((H, W), dtype=np.float32)
    #count_map = np.zeros((H, W), dtype=np.float32)
    pattern   = re.compile(r'_(\d{5})-(\d{5})_in\.png$')

    for img_path in img_paths:
        fname = os.path.basename(img_path)
        m = pattern.search(fname)
        if m is None:
            print(f"WARNING: 座標取得失敗: {fname}")
            continue
        x = int(m.group(1))
        y = int(m.group(2))

        pred_path = os.path.join(paths['patch_out_dir'], fname.replace('_in.png', '_pred.png'))
        if not os.path.exists(pred_path):
            continue

        pred  = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        y_end = min(y + TILE_SIZE, H)
        x_end = min(x + TILE_SIZE, W)
        merged[y:y_end, x:x_end] = np.maximum(merged[y:y_end, x:x_end],pred[:y_end-y, :x_end-x])

    #valid = count_map > 0
    #merged[valid] /= count_map[valid]
    merged_img = (merged * 255).astype(np.uint8)
    cv2.imwrite(paths['merged'], merged_img)
    print(f"つなぎ合わせ完了: {paths['merged']}")
    return merged_img


# -----------------------------------------------------------------
# Step 3: 可視化 + 評価指標計算
# -----------------------------------------------------------------
def blend_result(paths, region_name):
    print("=== Step 3: 可視化 ===")
    os.makedirs(paths['vis_dir'], exist_ok=True)

    img = cv2.imread(paths['in_img'], cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"元画像が見つかりません: {paths['in_img']}")

    gt_mask = cv2.imread(paths['gt_mask'], cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise FileNotFoundError(f"真値マスクが見つかりません: {paths['gt_mask']}")

    pred_mask = cv2.imread(paths['merged'], cv2.IMREAD_GRAYSCALE)
    if pred_mask is None:
        raise FileNotFoundError(f"推論結果が見つかりません: {paths['merged']}")

    H = min(img.shape[0], gt_mask.shape[0], pred_mask.shape[0])
    W = min(img.shape[1], gt_mask.shape[1], pred_mask.shape[1])
    img       = img[:H, :W]
    gt_mask   = gt_mask[:H, :W]
    pred_mask = pred_mask[:H, :W]

    # 膨張処理（可視化のみ）
    kernel       = np.ones((3, 3), np.uint8)
    gt_bin_vis   = cv2.dilate((gt_mask   > 0).astype(np.uint8), kernel, iterations=1).astype(bool)
    pred_bin_vis = cv2.dilate((pred_mask > 127).astype(np.uint8), kernel, iterations=1).astype(bool)

    gt_only   = gt_bin_vis & ~pred_bin_vis   # 赤: ラベルのみ
    pred_only = pred_bin_vis & ~gt_bin_vis   # シアン: 推論のみ
    both      = gt_bin_vis & pred_bin_vis    # 白: 一致

    color_mask = np.zeros_like(img)
    color_mask[gt_only]   = (0,   0,   255)  # 赤
    color_mask[pred_only] = (0,165, 255)    #orange
    color_mask[both]      = (255, 255, 0)   #sian
    #color_mask[both]      = (255, 255, 255)  # 白
    blend = cv2.addWeighted(img, 0.6, color_mask, 0.8, 0.0)
    ret = cv2.imwrite(paths['blend'], blend)
    print(f"可視化完了: {paths['blend']}" if ret else f"ERROR: 保存失敗: {paths['blend']}")

    # 評価指標（膨張なし）
    gt_bin   = (gt_mask > 0)
    pred_bin = (pred_mask > 127)

    TP = np.logical_and( gt_bin,  pred_bin).sum()
    FP = np.logical_and(~gt_bin,  pred_bin).sum()
    FN = np.logical_and( gt_bin, ~pred_bin).sum()
    TN = np.logical_and(~gt_bin, ~pred_bin).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"=== 評価結果（{region_name} / filamentクラス） ===")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  IoU       : {iou:.4f}")
    print(f"  F1 (Dice) : {f1:.4f}")

    with open(paths['metrics_csv'], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Region', 'Precision', 'Recall', 'IoU', 'F1', 'TP', 'FP', 'FN', 'TN'])
        writer.writerow([region_name, f'{precision:.4f}', f'{recall:.4f}',
                         f'{iou:.4f}', f'{f1:.4f}', TP, FP, FN, TN])
    print(f"メトリクス保存: {paths['metrics_csv']}")


# -----------------------------------------------------------------
# メイン
# -----------------------------------------------------------------
if __name__ == '__main__':
    # 引数からリージョンを指定
    # 例: python infer_and_blend.py orionB
    #     python infer_and_blend.py all
    arg = sys.argv[1] if len(sys.argv) > 1 else 'orionB'

    if arg == 'all':
        targets = list(REGION_MAP.keys())
    elif arg in REGION_MAP:
        targets = [arg]
    else:
        print(f"ERROR: 不明なリージョン '{arg}'")
        print(f"使用可能: {list(REGION_MAP.keys())} または all")
        sys.exit(1)

    print(f"対象リージョン: {targets}")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')

    for region_name in targets:
        fits_name = REGION_MAP[region_name]
        print(f"\n{'='*50}")
        print(f"処理中: {region_name} ({fits_name})")
        print(f"{'='*50}")
        paths = get_paths(fits_name)
        img_paths = run_inference(model, paths)
        merge_patches(img_paths, paths)
        blend_result(paths, region_name)

    print("\n=== 全処理完了 ===")
