#!/usr/bin/env python3
"""Region-wise evaluation for an MMSegmentation 1.x checkpoint.

For each region and for ALL_REGIONS, compute per-class Precision, Recall, IoU,
and Dice/F1 from the confusion matrix. Ground-truth label 255 is ignored.

The script can evaluate an intermediate checkpoint such as epoch_5.pth while
training continues, as long as the checkpoint file has already been written.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# =============================================================================
# USER SETTINGS: 基本的にここだけ変更
# =============================================================================

PROJECT_ROOT = Path(
    "/home1/fujiwara.r/PycharmProjects/"
    "mmsegmentation_research/mmsegmentation"
)

CONFIG_PATH = PROJECT_ROOT / (
    "configs/unet/"
    "unet_deeplabv3_hrf_3class_5regions_lshape.py"
)

WORK_DIR = PROJECT_ROOT / (
    "work_dirs/"
    "unet_deeplabv3_hrf_3class_5regions_lshape_balanced_v2"
)

# 途中のepochを明示的に評価する場合:
# CHECKPOINT_PATH = WORK_DIR / "epoch_5.pth"
# NoneならWORK_DIR内の最大epoch番号を自動選択します。
CHECKPOINT_PATH: Optional[Path] = "work_dirs/unet_deeplabv3_hrf_3class_5regions_lshape_balanced_v2/epoch_6.pth"

DATASET_ROOT = Path("/mnt/sp4t/fujiwara.r/split")

# "val" または "test"
EVAL_SPLIT = "val"

REGIONS = (
    "aquila",
    "orionb",
    "ic5146",
    "taurus",
    "polaris",
)

DEVICE = "cuda:0"

NUM_CLASSES = 3
IGNORE_LABEL = 255
CLASS_NAMES = (
    "background",
    "confirmed_filament",
    "candidate_filament",
)

OUTPUT_DIR = WORK_DIR / "region_evaluation"


# =============================================================================
# Utilities
# =============================================================================

def find_latest_epoch_checkpoint(work_dir: Path) -> Path:
    candidates: List[Tuple[int, Path]] = []
    for path in work_dir.glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(
            f"No epoch_*.pth checkpoint was found in: {work_dir}"
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_checkpoint() -> Path:
    if CHECKPOINT_PATH is not None:
        path = Path(CHECKPOINT_PATH)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    return find_latest_epoch_checkpoint(WORK_DIR)


def ensure_import_paths() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def image_to_mask_path(
    image_path: Path,
    image_region_root: Path,
    mask_region_root: Path,
) -> Path:
    relative = image_path.relative_to(image_region_root)
    if not relative.name.endswith("_in.png"):
        raise ValueError(f"Unexpected input filename: {image_path}")
    mask_name = relative.name[:-len("_in.png")] + "_out.png"
    return mask_region_root / relative.parent / mask_name


def collect_pairs(
    dataset_root: Path,
    split: str,
    region: str,
) -> List[Tuple[Path, Path]]:
    image_root = dataset_root / split / "in" / region
    mask_root = dataset_root / split / "out" / region

    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_root}")

    image_paths = sorted(image_root.rglob("*_in.png"))
    if not image_paths:
        raise RuntimeError(f"No *_in.png files found in: {image_root}")

    pairs: List[Tuple[Path, Path]] = []
    for image_path in image_paths:
        mask_path = image_to_mask_path(
            image_path, image_root, mask_root
        )
        if not mask_path.is_file():
            raise FileNotFoundError(
                "Corresponding mask not found:\n"
                f"  image: {image_path}\n"
                f"  mask:  {mask_path}"
            )
        pairs.append((image_path, mask_path))

    return pairs


def read_gt_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))

    if mask.ndim == 3:
        if (
            mask.shape[2] >= 3
            and np.array_equal(mask[..., 0], mask[..., 1])
            and np.array_equal(mask[..., 0], mask[..., 2])
        ):
            mask = mask[..., 0]
        else:
            raise ValueError(
                f"Mask must be a label image: {path}, shape={mask.shape}"
            )

    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(
            f"Mask must be 2-D: {path}, shape={mask.shape}"
        )

    return mask.astype(np.int64, copy=False)


def extract_prediction(result) -> np.ndarray:
    if not hasattr(result, "pred_sem_seg"):
        raise AttributeError(
            "Inference result has no pred_sem_seg; MMSegmentation 1.x is expected."
        )

    pred = result.pred_sem_seg.data
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    else:
        pred = np.asarray(pred)

    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"Prediction must be 2-D: shape={pred.shape}")

    return pred.astype(np.int64, copy=False)


def update_confusion(
    confusion: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
) -> None:
    if gt.shape != pred.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: pred={pred.shape}, gt={gt.shape}"
        )

    valid = gt != IGNORE_LABEL
    gt_valid = gt[valid]
    pred_valid = pred[valid]

    valid_range = (
        (gt_valid >= 0)
        & (gt_valid < NUM_CLASSES)
        & (pred_valid >= 0)
        & (pred_valid < NUM_CLASSES)
    )
    gt_valid = gt_valid[valid_range]
    pred_valid = pred_valid[valid_range]

    encoded = NUM_CLASSES * gt_valid + pred_valid
    counts = np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    )
    confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES)


def safe_divide(a: float, b: float) -> float:
    return float("nan") if b == 0 else float(a / b)


def metrics_from_confusion(confusion: np.ndarray) -> Dict[str, object]:
    tp = np.diag(confusion).astype(np.float64)
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    fp = pred_count - tp
    fn = gt_count - tp

    rows: List[Dict[str, object]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        precision = safe_divide(tp[class_id], tp[class_id] + fp[class_id])
        recall = safe_divide(tp[class_id], tp[class_id] + fn[class_id])
        iou = safe_divide(
            tp[class_id],
            tp[class_id] + fp[class_id] + fn[class_id],
        )
        dice = safe_divide(
            2.0 * tp[class_id],
            2.0 * tp[class_id] + fp[class_id] + fn[class_id],
        )

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_pixels": int(gt_count[class_id]),
                "pred_pixels": int(pred_count[class_id]),
                "tp": int(tp[class_id]),
                "fp": int(fp[class_id]),
                "fn": int(fn[class_id]),
                "precision": precision,
                "recall": recall,
                "iou": iou,
                "dice_f1": dice,
            }
        )

    total = float(confusion.sum())
    overall_accuracy = safe_divide(float(tp.sum()), total)

    return {
        "overall_accuracy": overall_accuracy,
        "mean_precision": float(np.nanmean([r["precision"] for r in rows])),
        "mean_recall": float(np.nanmean([r["recall"] for r in rows])),
        "mean_iou": float(np.nanmean([r["iou"] for r in rows])),
        "mean_dice_f1": float(np.nanmean([r["dice_f1"] for r in rows])),
        "classes": rows,
    }


def pct(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value * 100:.2f}"


def print_table(
    region: str,
    patch_count: int,
    metrics: Dict[str, object],
) -> None:
    print()
    print("=" * 94)
    print(f"Region: {region}    patches={patch_count}")
    print("=" * 94)
    print(
        f"{'Class':22s} {'Precision':>10s} {'Recall':>10s} "
        f"{'IoU':>10s} {'Dice/F1':>10s} {'GT pixels':>14s}"
    )
    print("-" * 94)

    for row in metrics["classes"]:
        print(
            f"{row['class_name']:22s} "
            f"{pct(row['precision']):>10s} "
            f"{pct(row['recall']):>10s} "
            f"{pct(row['iou']):>10s} "
            f"{pct(row['dice_f1']):>10s} "
            f"{row['gt_pixels']:14d}"
        )

    print("-" * 94)
    print(
        f"aAcc={pct(metrics['overall_accuracy'])}%  "
        f"mPrecision={pct(metrics['mean_precision'])}%  "
        f"mRecall={pct(metrics['mean_recall'])}%  "
        f"mIoU={pct(metrics['mean_iou'])}%  "
        f"mDice/F1={pct(metrics['mean_dice_f1'])}%"
    )


def save_csv(
    path: Path,
    results: Dict[str, Dict[str, object]],
) -> None:
    fields = [
        "region", "patch_count", "class_id", "class_name",
        "gt_pixels", "pred_pixels", "tp", "fp", "fn",
        "precision", "recall", "iou", "dice_f1",
        "overall_accuracy", "mean_precision", "mean_recall",
        "mean_iou", "mean_dice_f1",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for region, region_result in results.items():
            metrics = region_result["metrics"]
            for row in metrics["classes"]:
                writer.writerow(
                    {
                        "region": region,
                        "patch_count": region_result["patch_count"],
                        **row,
                        "overall_accuracy": metrics["overall_accuracy"],
                        "mean_precision": metrics["mean_precision"],
                        "mean_recall": metrics["mean_recall"],
                        "mean_iou": metrics["mean_iou"],
                        "mean_dice_f1": metrics["mean_dice_f1"],
                    }
                )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if EVAL_SPLIT not in {"val", "test"}:
        raise ValueError("EVAL_SPLIT must be 'val' or 'test'.")

    ensure_import_paths()

    from mmseg.apis import inference_model, init_model

    checkpoint = resolve_checkpoint()
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    print("=" * 94)
    print("Region-wise MMSegmentation evaluation")
    print(f"Config:     {CONFIG_PATH}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Split:      {EVAL_SPLIT}")
    print(f"Device:     {DEVICE}")
    print("=" * 94)

    model = init_model(
        str(CONFIG_PATH),
        str(checkpoint),
        device=DEVICE,
    )
    
     # ここを追加
    model.cfg.test_pipeline = [
       dict(type='LoadImageFromFile'),
       dict(type='PackSegInputs'),
    ]


    results: Dict[str, Dict[str, object]] = {}
    overall_confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )
    overall_patch_count = 0

    for region in REGIONS:
        pairs = collect_pairs(DATASET_ROOT, EVAL_SPLIT, region)
        confusion = np.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=np.int64,
        )

        print(f"\n[{region}] evaluating {len(pairs)} patches...")

        for index, (image_path, mask_path) in enumerate(pairs, start=1):
            result = inference_model(model, str(image_path))
            pred = extract_prediction(result)
            gt = read_gt_mask(mask_path)
            update_confusion(confusion, gt, pred)

            if index % 50 == 0 or index == len(pairs):
                print(f"  {index}/{len(pairs)}", flush=True)

        metrics = metrics_from_confusion(confusion)
        results[region] = {
            "patch_count": len(pairs),
            "confusion_matrix": confusion.tolist(),
            "metrics": metrics,
        }

        overall_confusion += confusion
        overall_patch_count += len(pairs)
        print_table(region, len(pairs), metrics)

    overall_metrics = metrics_from_confusion(overall_confusion)
    results["ALL_REGIONS"] = {
        "patch_count": overall_patch_count,
        "confusion_matrix": overall_confusion.tolist(),
        "metrics": overall_metrics,
    }
    print_table("ALL_REGIONS", overall_patch_count, overall_metrics)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = checkpoint.stem
    json_path = OUTPUT_DIR / f"{stem}_{EVAL_SPLIT}_region_metrics.json"
    csv_path = OUTPUT_DIR / f"{stem}_{EVAL_SPLIT}_region_metrics.csv"

    payload = {
        "config": str(CONFIG_PATH),
        "checkpoint": str(checkpoint),
        "eval_split": EVAL_SPLIT,
        "dataset_root": str(DATASET_ROOT),
        "class_names": list(CLASS_NAMES),
        "ignore_label": IGNORE_LABEL,
        "metric_definitions": {
            "precision": "TP/(TP+FP)",
            "recall": "TP/(TP+FN)",
            "iou": "TP/(TP+FP+FN)",
            "dice_f1": "2TP/(2TP+FP+FN)",
        },
        "results": results,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=True)

    save_csv(csv_path, results)

    print("\n" + "=" * 94)
    print("Saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print("=" * 94)


if __name__ == "__main__":
    main()
