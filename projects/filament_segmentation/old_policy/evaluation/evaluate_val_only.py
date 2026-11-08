"""
val限定 評価スクリプト(train・除外パッチを含まない、正しい検証性能の算出)

infer_and_blend.py はフルマップ再構成方式のため、
in_dir(雲全体の全パッチ)を対象にしてしまい、
train用パッチ・バッファ除外パッチ・背景のみパッチまで含めて評価してしまう
という問題があった。

このスクリプトは、split/val_fixed.txt に記載された
「その雲(region)に属するvalパッチ」のみに絞り込み、
パッチ単体でTP/FP/FN/TNを積算して評価する。

これにより、モデルが学習時に見ていない(あるいは
少なくともtrainには使われていない)パッチのみで、
正しい検証性能を算出できる。

使い方:
  # 特定のseedのcheckpointに対して評価
  python evaluate_val_only.py --checkpoint work_dirs/reproducibility_check/seed_2/iter_40000.pth

  # 5.3節などの通常学習の結果を評価
  python evaluate_val_only.py --checkpoint work_dirs/unet_deeplabv3_hgbs/iter_40000.pth
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2

from mmseg.apis import init_model, inference_model

# ==================================================
# 設定
# ==================================================

CONFIG_FILE = 'configs/unet/unet_deeplabv3_my_dataset.py'
SPLIT_ROOT = '/mnt/sp4t/fujiwara.r/split'
VAL_LIST_PATH = os.path.join(SPLIT_ROOT, 'val_fixed.txt')

# fits_name(FilFinderの出力ディレクトリ名) <-> region略称の対応
# val_fixed.txt の中に、この文字列を含む行がその雲に属すると判定する
REGION_MAP = {
    'aquila': 'HGBS_aquilaM2_hires_column_density_map',
    'ic5146': 'HGBS_ic5146_hires_column_density_map',
}

FILAMENT_LABEL_VALUE = 1   # split/val/out のマスクにおけるフィラメントのピクセル値
PRED_FILAMENT_CLASS = 1    # モデル出力のフィラメントクラスindex

OUTPUT_DIR = os.path.join(SPLIT_ROOT, 'eval_val_only')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# util
# ==================================================

def load_val_stems_by_region():
    """
    val_fixed.txt (例: "tiles_003/HGBS_aquilaM2_..._01234-05678")
    を読み込み、region略称ごとに該当する行(スタブ名)を振り分ける。
    """
    with open(VAL_LIST_PATH, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    region_stems = {region: [] for region in REGION_MAP}

    for line in lines:
        for region, fits_name in REGION_MAP.items():
            if fits_name in line:
                region_stems[region].append(line)
                break

    for region, stems in region_stems.items():
        print(f"[{region}] val_fixed.txt 中の該当パッチ数: {len(stems)}")

    return region_stems


def stem_to_paths(stem):
    """
    stem 例: "tiles_003/HGBS_aquilaM2_hires_column_density_map_01234-05678"
    -> in_path, out_path を組み立てる
    """
    in_path = os.path.join(SPLIT_ROOT, 'val', 'in', stem + '_in.png')
    out_path = os.path.join(SPLIT_ROOT, 'val', 'out', stem + '_out.png')
    return in_path, out_path


def load_gt_mask(path, label_value=FILAMENT_LABEL_VALUE):
    mask = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask == label_value).astype(np.uint8)


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0
          else np.nan)
    return precision, recall, iou, f1


# ==================================================
# メイン評価処理
# ==================================================

def evaluate_region(model, region, stems):
    """
    region(例: 'aquila')に属するvalパッチのみを対象に推論し、
    TP/FP/FNをパッチ全体で積算して1回だけ指標を算出する。
    """
    agg_tp = agg_fp = agg_fn = 0
    n_valid = 0
    n_missing = 0

    for stem in stems:
        in_path, out_path = stem_to_paths(stem)

        if not os.path.exists(in_path) or not os.path.exists(out_path):
            n_missing += 1
            continue

        result = inference_model(model, in_path)
        pred_seg = result.pred_sem_seg.data.cpu().numpy().squeeze()
        pred_mask = (pred_seg == PRED_FILAMENT_CLASS).astype(bool)

        gt_mask = load_gt_mask(out_path).astype(bool)

        agg_tp += np.logical_and(pred_mask, gt_mask).sum()
        agg_fp += np.logical_and(pred_mask, ~gt_mask).sum()
        agg_fn += np.logical_and(~pred_mask, gt_mask).sum()
        n_valid += 1

    if n_missing > 0:
        print(f"  WARNING: {n_missing}件のパッチファイルが見つかりませんでした"
              f"(split/val 生成時と in_dir 構成が変わった可能性があります)")

    precision, recall, iou, f1 = compute_metrics(agg_tp, agg_fp, agg_fn)

    return dict(
        region=region,
        n_patches=n_valid,
        tp=int(agg_tp), fp=int(agg_fp), fn=int(agg_fn),
        precision=precision, recall=recall, iou=iou, f1=f1,
    )


# ==================================================
# エントリポイント
# ==================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                         help='評価したいモデルのcheckpointパス(.pth)')
    parser.add_argument('--config', default=CONFIG_FILE)
    parser.add_argument('--tag', default='',
                         help='出力CSVファイル名に付与するタグ(例: seed2 など)')
    args = parser.parse_args()

    print(f"config: {args.config}")
    print(f"checkpoint: {args.checkpoint}")

    model = init_model(args.config, args.checkpoint, device='cuda:0')

    region_stems = load_val_stems_by_region()

    rows = []
    for region, stems in region_stems.items():
        if len(stems) == 0:
            print(f"[{region}] valパッチが0件のためスキップ")
            continue

        print(f"\n=== 評価中: {region} (val限定, n={len(stems)}) ===")
        m = evaluate_region(model, region, stems)
        rows.append(m)

        print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
              f"IoU={m['iou']:.4f}  F1={m['f1']:.4f}  (n_patches={m['n_patches']})")

    df = pd.DataFrame(rows)
    tag_suffix = f"_{args.tag}" if args.tag else ""
    csv_path = os.path.join(OUTPUT_DIR, f"val_only_metrics{tag_suffix}.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== val限定 評価結果まとめ(train・除外パッチを含まない) ===")
    print(df[["region", "n_patches", "precision", "recall", "iou", "f1"]].to_string(index=False))
    print(f"\n保存: {csv_path}")
