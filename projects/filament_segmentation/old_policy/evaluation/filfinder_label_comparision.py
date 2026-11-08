"""
フィラメント検出モデルの評価スクリプト

- 学習済みmmsegモデルでAquila / IC5146 / OrionBに推論を実施
- FilFinderラベル(GT)と予測結果を比較し、
  Precision / Recall / IoU / F1(Dice) を領域ごとに算出
- 赤(見逃し/FN)・オレンジ(過検出/FP)・シアン(一致/TP)の重ね合わせ画像を生成

出力:
eval_results/
├── metrics_summary.csv          # 領域ごとの指標一覧
├── Aquila/
│   ├── overlay_tiles_000_xxxx.png
│   └── ...
├── IC5146/
│   └── ...
└── OrionB/
    └── ...
"""

import os
import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from mmseg.apis import init_model, inference_model

# ==================================================
# 設定
# ==================================================

root_dir = "/mnt/sp4t/fujiwara.r"

CONFIG_PATH = "configs/unet/unet_deeplabv3_my_dataset.py"          # 学習に使ったconfig
CHECKPOINT_PATH = "work_dirs/reproducibility_check/seed_1/iter_40000.pth"  # 学習済み重み

FILAMENT_LABEL_VALUE = 1   # GTマスクのフィラメントクラス値
PRED_FILAMENT_CLASS = 1    # モデル出力のフィラメントクラスindex

# 評価対象領域: 領域名 -> (in画像ディレクトリ, out(GT)画像ディレクトリ)
# train/valに使ったAquila・IC5146と、未学習のOrionBをすべて評価する
EVAL_REGIONS = {
    "Aquila": {
        "in_dir":  os.path.join(root_dir, "HGBS_aquilaM2_hires_column_density_map", "in"),
        "out_dir": os.path.join(root_dir, "HGBS_aquilaM2_hires_column_density_map", "out"),
    },
    "IC5146": {
        "in_dir":  os.path.join(root_dir, "HGBS_ic5146_hires_column_density_map", "in"),
        "out_dir": os.path.join(root_dir, "HGBS_ic5146_hires_column_density_map", "out"),
    },
    "OrionB": {
        "in_dir":  os.path.join(root_dir, "HGBS_orionB_hires_column_density_map", "in"),
        "out_dir": os.path.join(root_dir, "HGBS_orionB_hires_column_density_map", "out"),
    },
}

# 可視化画像を保存する枚数の上限(全パッチだと膨大になるため)
MAX_VIS_PER_REGION = 30

output_dir = os.path.join(root_dir, "eval_results")
os.makedirs(output_dir, exist_ok=True)


# ==================================================
# util
# ==================================================

def load_gt_mask(path, label_value=FILAMENT_LABEL_VALUE):
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask == label_value).astype(np.uint8)


def collect_tile_paths(in_dir, out_dir):
    """
    in_dir配下の全 *_in.png と、対応する out_dir配下の *_out.png のペアを集める。
    """
    pairs = []
    for in_path in sorted(glob.glob(os.path.join(in_dir, "tiles_*", "*_in.png"))):
        out_path = in_path.replace(in_dir, out_dir).replace("_in.png", "_out.png")
        if os.path.exists(out_path):
            pairs.append((in_path, out_path))
    return pairs


def compute_metrics(pred_mask, gt_mask):
    """
    pred_mask, gt_mask: 0/1のuint8配列(同じshape)
    戻り値: dict(tp, fp, fn, tn, precision, recall, iou, f1)
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if (precision is not np.nan and recall is not np.nan and (precision + recall) > 0)
          else np.nan)

    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, iou=iou, f1=f1)


def make_overlay(input_img, pred_mask, gt_mask):
    """
    赤:見逃し(GTにあるがpredにない, FN)
    オレンジ:過検出(predにあるがGTにない, FP)
    シアン:一致(TP)
    """
    if input_img.ndim == 2:
        base = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    else:
        base = input_img.copy()

    overlay = base.copy()

    tp_mask = np.logical_and(pred_mask, gt_mask)
    fp_mask = np.logical_and(pred_mask, ~gt_mask.astype(bool))
    fn_mask = np.logical_and(~pred_mask.astype(bool), gt_mask)

    overlay[fn_mask] = (0, 0, 255)     # 赤(BGR): 見逃し
    overlay[fp_mask] = (0, 165, 255)   # オレンジ(BGR): 過検出
    overlay[tp_mask] = (255, 255, 0)   # シアン(BGR): 一致

    return overlay


# ==================================================
# モデルロード
# ==================================================

print("loading model...")
model = init_model(CONFIG_PATH, CHECKPOINT_PATH, device="cuda:0")


# ==================================================
# 領域ごとに評価
# ==================================================

summary_rows = []

for region_name, dirs in EVAL_REGIONS.items():
    in_dir, out_dir = dirs["in_dir"], dirs["out_dir"]

    if not os.path.isdir(in_dir):
        print(f"[{region_name}] SKIP: {in_dir} が存在しません")
        continue

    pairs = collect_tile_paths(in_dir, out_dir)
    print(f"[{region_name}] evaluating {len(pairs)} tiles...")

    region_vis_dir = os.path.join(output_dir, region_name)
    os.makedirs(region_vis_dir, exist_ok=True)

    agg_tp = agg_fp = agg_fn = agg_tn = 0
    per_tile_metrics = []

    for idx, (in_path, out_path) in enumerate(pairs):
        # 推論
        result = inference_model(model, in_path)
        pred_seg = result.pred_sem_seg.data.cpu().numpy().squeeze()  # (H, W) クラスindex
        pred_mask = (pred_seg == PRED_FILAMENT_CLASS).astype(np.uint8)

        gt_mask = load_gt_mask(out_path)

        m = compute_metrics(pred_mask, gt_mask)
        per_tile_metrics.append(m)

        agg_tp += m["tp"]
        agg_fp += m["fp"]
        agg_fn += m["fn"]
        agg_tn += m["tn"]

        # 可視化(上限枚数まで)
        if idx < MAX_VIS_PER_REGION:
            input_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            overlay = make_overlay(input_img, pred_mask, gt_mask)
            vis_name = os.path.basename(in_path).replace("_in.png", "_overlay.png")
            cv2.imwrite(os.path.join(region_vis_dir, vis_name), overlay)

    # 領域全体で集計(タイル単位平均ではなく、ピクセル数を積算してから算出)
    precision = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else np.nan
    recall = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else np.nan
    iou = agg_tp / (agg_tp + agg_fp + agg_fn) if (agg_tp + agg_fp + agg_fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0
          else np.nan)

    print(f"[{region_name}] Precision={precision:.4f} Recall={recall:.4f} "
          f"IoU={iou:.4f} F1={f1:.4f} (n_tiles={len(pairs)})")

    summary_rows.append(dict(
        region=region_name,
        n_tiles=len(pairs),
        tp=agg_tp, fp=agg_fp, fn=agg_fn, tn=agg_tn,
        precision=precision, recall=recall, iou=iou, f1=f1,
    ))

# ==================================================
# サマリ保存
# ==================================================

summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(output_dir, "metrics_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

print("\n=== 評価結果まとめ ===")
print(summary_df[["region", "n_tiles", "precision", "recall", "iou", "f1"]]
      .to_string(index=False))
print(f"\nsaved: {summary_csv_path}")
