#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_precision_recall_f1_shared_aspp.py

Current model:
    HRF-pretrained DeepLabV3 (UNet Backbone)
    + fully shared ASPP
    + final 1x1 Conv dual heads

Outputs for each region and class:
    Precision
    Recall
    F1
    Threshold
    TP / FP / FN / TN

Default:
    validation split
    threshold = best F1 threshold independently for each region/class

Example:
    python tools/analysis_tools/evaluate_precision_recall_f1_shared_aspp.py \
        --checkpoint work_dirs/hrf_deeplabv3_unet_shared_aspp_dual_bce_dice_v2/epoch_17.pth
"""

from pathlib import Path
import argparse
import csv
import sys

import numpy as np
from PIL import Image


DEFAULT_PROJECT_ROOT = Path(
    "/home1/fujiwara.r/PycharmProjects/mmsegmentation_research/mmsegmentation"
)
DEFAULT_DATASET_ROOT = Path("/mnt/sp4t/fujiwara.r/split_v2")
DEFAULT_CONFIG = (
    DEFAULT_PROJECT_ROOT
    / "configs/unet/hrf_deeplabv3_unet_shared_aspp_dual_bce_dice_v2.py"
)
DEFAULT_CHECKPOINT = (
    DEFAULT_PROJECT_ROOT
    / "work_dirs/hrf_deeplabv3_unet_shared_aspp_dual_bce_dice_v2/epoch_17.pth"
)
DEFAULT_OUTPUT = (
    DEFAULT_DATASET_ROOT
    / "shared_aspp_dual_head_precision_recall_f1_validation.csv"
)
DEFAULT_REGIONS = ["aquila", "orionb", "ic5146", "taurus", "polaris"]

BACKGROUND_LABEL = 0
CLASS1_LABEL = 1
CLASS2_LABEL = 2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)

    parser.add_argument(
        "--threshold-mode",
        choices=("best-f1", "fixed"),
        default="best-f1",
    )
    parser.add_argument("--class1-threshold", type=float, default=0.5)
    parser.add_argument("--class2-threshold", type=float, default=0.3)

    args = parser.parse_args()

    for name in ("class1_threshold", "class2_threshold"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    return args


def collect_pairs(dataset_root: Path, split: str, region: str):
    inp = dataset_root / split / "in" / region
    out = dataset_root / split / "out" / region

    if not inp.is_dir():
        raise FileNotFoundError(f"Input directory not found: {inp}")
    if not out.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {out}")

    pairs = []

    for image_path in sorted(inp.rglob("*_in.png")):
        rel = image_path.relative_to(inp)
        mask_path = out / rel.parent / rel.name.replace("_in.png", "_out.png")

        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)

        pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No patches found: split={split}, region={region}")

    return pairs


def read_mask(path: Path):
    arr = np.squeeze(np.asarray(Image.open(path)))

    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D: {path}, shape={arr.shape}")

    return arr.astype(np.uint8)


def _to_numpy(x):
    if hasattr(x, "data") and not isinstance(x, np.ndarray):
        try:
            x = x.data
        except Exception:
            pass

    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()

    return np.asarray(x)


def extract_probability(result, class_id: int):
    if class_id == 1:
        names = ("class1_prob", "confirmed_prob")
    elif class_id == 2:
        names = ("class2_prob", "candidate_prob")
    else:
        raise ValueError(class_id)

    for name in names:
        if hasattr(result, name):
            arr = np.squeeze(_to_numpy(getattr(result, name)))
            if arr.ndim == 2:
                return arr.astype(np.float32)

    if isinstance(result, dict):
        for name in names:
            if name in result:
                arr = np.squeeze(_to_numpy(result[name]))
                if arr.ndim == 2:
                    return arr.astype(np.float32)

    attrs = [
        name
        for name in dir(result)
        if ("class1" in name or "class2" in name or "prob" in name)
        and not name.startswith("_")
    ]

    raise AttributeError(
        f"Class {class_id} probability not found. Available={attrs}"
    )


def verify_model(model):
    required = ("decode_head", "class1_classifier", "class2_classifier")
    missing = [name for name in required if not hasattr(model, name)]

    if missing:
        raise RuntimeError(
            f"Wrong model implementation. Missing={missing}. "
            "Use the shared-ASPP dual-head model."
        )


def confusion_from_threshold(y_true, y_score, threshold):
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)

    pred = y_score >= threshold
    pos = y_true == 1
    neg = y_true == 0

    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & neg))
    fn = int(np.count_nonzero((~pred) & pos))
    tn = int(np.count_nonzero((~pred) & neg))

    return tp, fp, fn, tn


def metric_values(tp, fp, fn):
    eps = 1e-12

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return float(precision), float(recall), float(f1)


def find_best_f1_threshold(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)

    if not np.any(y_true == 1):
        raise ValueError("No positive pixels.")

    order = np.argsort(-y_score, kind="mergesort")
    yt = y_true[order]
    ys = y_score[order]

    tp = np.cumsum(yt == 1, dtype=np.float64)
    fp = np.cumsum(yt == 0, dtype=np.float64)

    total_pos = float(np.sum(y_true == 1))
    fn = total_pos - tp

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / total_pos
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)

    distinct = np.where(np.diff(ys) != 0)[0]
    idx = np.r_[distinct, ys.size - 1]

    best_local = int(np.nanargmax(f1[idx]))
    best_index = int(idx[best_local])

    return float(ys[best_index])


def evaluate_region(
    model,
    inference_model,
    dataset_root,
    split,
    region,
    threshold_mode,
    class1_threshold,
    class2_threshold,
):
    pairs = collect_pairs(dataset_root, split, region)

    c1_true, c1_score = [], []
    c2_true, c2_score = [], []

    for i, (image_path, mask_path) in enumerate(pairs, 1):
        gt = read_mask(mask_path)
        result = inference_model(model, str(image_path))

        p1 = extract_probability(result, 1)
        p2 = extract_probability(result, 2)

        if p1.shape != gt.shape:
            raise ValueError(
                f"Class 1 shape mismatch: {image_path}, prob={p1.shape}, gt={gt.shape}"
            )
        if p2.shape != gt.shape:
            raise ValueError(
                f"Class 2 shape mismatch: {image_path}, prob={p2.shape}, gt={gt.shape}"
            )

        valid1 = (gt == BACKGROUND_LABEL) | (gt == CLASS1_LABEL)
        c1_true.append((gt[valid1] == CLASS1_LABEL).astype(np.uint8))
        c1_score.append(p1[valid1].astype(np.float32))

        valid2 = (gt == BACKGROUND_LABEL) | (gt == CLASS2_LABEL)
        c2_true.append((gt[valid2] == CLASS2_LABEL).astype(np.uint8))
        c2_score.append(p2[valid2].astype(np.float32))

        if i % 50 == 0 or i == len(pairs):
            print(f"  {region}: {i}/{len(pairs)}")

    y1 = np.concatenate(c1_true)
    s1 = np.concatenate(c1_score)
    y2 = np.concatenate(c2_true)
    s2 = np.concatenate(c2_score)

    if threshold_mode == "best-f1":
        th1 = find_best_f1_threshold(y1, s1)
        th2 = find_best_f1_threshold(y2, s2)
    else:
        th1 = float(class1_threshold)
        th2 = float(class2_threshold)

    rows = []

    for class_name, y_true, y_score, threshold in (
        ("Class 1", y1, s1, th1),
        ("Class 2", y2, s2, th2),
    ):
        tp, fp, fn, tn = confusion_from_threshold(
            y_true, y_score, threshold
        )

        precision, recall, f1 = metric_values(tp, fp, fn)

        rows.append(
            dict(
                region=region,
                class_name=class_name,
                threshold=threshold,
                precision_percent=precision * 100.0,
                recall_percent=recall * 100.0,
                f1_percent=f1 * 100.0,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                gt_positive_pixels=int(np.sum(y_true == 1)),
                predicted_positive_pixels=int(np.sum(y_score >= threshold)),
                max_probability=float(np.max(y_score)),
            )
        )

    return rows


def save_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "region",
        "class_name",
        "threshold",
        "precision_percent",
        "recall_percent",
        "f1_percent",
        "tp",
        "fp",
        "fn",
        "tn",
        "gt_positive_pixels",
        "predicted_positive_pixels",
        "max_probability",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in fields}
            for row in rows
        )


def print_table(rows):
    print()
    print("=" * 92)
    print(
        f"{'Region':10s} "
        f"{'Class':8s} "
        f"{'Threshold':>10s} "
        f"{'Precision':>12s} "
        f"{'Recall':>10s} "
        f"{'F1':>10s}"
    )
    print("-" * 92)

    for row in rows:
        print(
            f"{row['region']:10s} "
            f"{row['class_name']:8s} "
            f"{row['threshold']:10.6f} "
            f"{row['precision_percent']:11.2f}% "
            f"{row['recall_percent']:9.2f}% "
            f"{row['f1_percent']:9.2f}%"
        )

    print("=" * 92)


def main():
    args = parse_args()

    if str(args.project_root) not in sys.path:
        sys.path.insert(0, str(args.project_root))

    from mmengine.config import Config
    from mmseg.apis import init_model, inference_model

    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    print("Config         :", args.config)
    print("Checkpoint     :", args.checkpoint)
    print("Dataset root   :", args.dataset_root)
    print("Split          :", args.split)
    print("Threshold mode :", args.threshold_mode)

    cfg = Config.fromfile(str(args.config))

    model = init_model(
        cfg,
        str(args.checkpoint),
        device=args.device,
    )

    verify_model(model)

    model.cfg.test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="PackSegInputs"),
    ]

    all_rows = []

    for region in args.regions:
        print("\n" + "=" * 80)
        print("Region:", region)

        rows = evaluate_region(
            model=model,
            inference_model=inference_model,
            dataset_root=args.dataset_root,
            split=args.split,
            region=region,
            threshold_mode=args.threshold_mode,
            class1_threshold=args.class1_threshold,
            class2_threshold=args.class2_threshold,
        )

        all_rows.extend(rows)

    print_table(all_rows)
    save_csv(all_rows, args.output)

    print()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
