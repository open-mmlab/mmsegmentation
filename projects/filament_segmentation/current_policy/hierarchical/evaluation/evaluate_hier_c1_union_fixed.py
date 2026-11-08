#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

from mmseg.apis import init_model

from hier_c1_union_common import (
    gather_split,
    best_f1_threshold,
    binary_metrics,
    auprc,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--model-name",
        required=True,
    )

    parser.add_argument(
        "--data-root",
        default="/mnt/sp4t/fujiwara.r/split_v2",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--out-csv",
        required=True,
    )

    args = parser.parse_args()

    print(
        "[INFO] model =",
        args.model_name,
    )

    print(
        "[INFO] config =",
        args.config,
    )

    print(
        "[INFO] checkpoint =",
        args.checkpoint,
    )

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
    )

    # ========================================================
    # Validation
    # ========================================================

    print(
        "\n===== VALIDATION ====="
    )

    val = gather_split(
        model,
        args.data_root,
        "val",
    )

    # ========================================================
    # Test
    # ========================================================

    print(
        "\n===== TEST ====="
    )

    test = gather_split(
        model,
        args.data_root,
        "test",
    )

    out_csv = Path(
        args.out_csv
    )

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_csv = (
        out_csv.with_name(
            out_csv.stem
            + "_thresholds.csv"
        )
    )

    preferred_regions = [
        "Aquila",
        "IC5146",
        "OrionB",
        "Taurus",
        "Polaris",
        "ALL",
    ]

    regions = [
        region
        for region
        in preferred_regions
        if (
            region in val
            and region in test
        )
    ]

    if len(regions) == 0:
        raise RuntimeError(
            "No common regions found "
            "between validation and test."
        )

    # ========================================================
    # Validation thresholds
    # ========================================================

    thresholds = {}
    threshold_rows = []

    print(
        "\n===== BEST VALIDATION THRESHOLDS ====="
    )

    for region in regions:

        (
            class1_threshold,
            val_class1_precision,
            val_class1_recall,
            val_class1_f1,
        ) = best_f1_threshold(
            val[
                region
            ][
                "y1"
            ],
            val[
                region
            ][
                "p1"
            ],
        )

        (
            union_threshold,
            val_union_precision,
            val_union_recall,
            val_union_f1,
        ) = best_f1_threshold(
            val[
                region
            ][
                "yu"
            ],
            val[
                region
            ][
                "pu"
            ],
        )

        thresholds[
            region
        ] = (
            class1_threshold,
            union_threshold,
        )

        threshold_rows.append({
            "model":
                args.model_name,

            "region":
                region,

            "class1_threshold":
                class1_threshold,

            "union_threshold":
                union_threshold,

            "val_class1_precision":
                val_class1_precision,

            "val_class1_recall":
                val_class1_recall,

            "val_class1_f1":
                val_class1_f1,

            "val_union_precision":
                val_union_precision,

            "val_union_recall":
                val_union_recall,

            "val_union_f1":
                val_union_f1,
        })

        print(
            "{:8s}  "
            "C1 th={:.4f} F1={:.4f} | "
            "Union th={:.4f} F1={:.4f}".format(
                region,
                class1_threshold,
                val_class1_f1,
                union_threshold,
                val_union_f1,
            )
        )

    with threshold_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                threshold_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            threshold_rows
        )

    # ========================================================
    # Test metrics
    # ========================================================

    rows = []

    print(
        "\n===== TEST RESULTS ====="
    )

    for region in regions:

        (
            class1_threshold,
            union_threshold,
        ) = thresholds[
            region
        ]

        (
            class1_precision,
            class1_recall,
            class1_f1,
            class1_tp,
            class1_fp,
            class1_fn,
        ) = binary_metrics(
            test[
                region
            ][
                "y1"
            ],
            test[
                region
            ][
                "p1"
            ],
            class1_threshold,
        )

        (
            union_precision,
            union_recall,
            union_f1,
            union_tp,
            union_fp,
            union_fn,
        ) = binary_metrics(
            test[
                region
            ][
                "yu"
            ],
            test[
                region
            ][
                "pu"
            ],
            union_threshold,
        )

        class1_auprc = auprc(
            test[
                region
            ][
                "y1"
            ],
            test[
                region
            ][
                "p1"
            ],
        )

        union_auprc = auprc(
            test[
                region
            ][
                "yu"
            ],
            test[
                region
            ][
                "pu"
            ],
        )

        mean_f1 = (
            class1_f1
            + union_f1
        ) / 2.0

        row = {
            "model":
                args.model_name,

            "region":
                region,

            "class1_threshold":
                class1_threshold,

            "class1_precision":
                class1_precision,

            "class1_recall":
                class1_recall,

            "class1_f1":
                class1_f1,

            "class1_auprc":
                class1_auprc,

            "class1_tp":
                class1_tp,

            "class1_fp":
                class1_fp,

            "class1_fn":
                class1_fn,

            "union_threshold":
                union_threshold,

            "union_precision":
                union_precision,

            "union_recall":
                union_recall,

            "union_f1":
                union_f1,

            "union_auprc":
                union_auprc,

            "union_tp":
                union_tp,

            "union_fp":
                union_fp,

            "union_fn":
                union_fn,

            "mean_f1":
                mean_f1,
        }

        rows.append(
            row
        )

        print(
            "{:8s}  "
            "C1 P={:.4f} R={:.4f} F1={:.4f} AUPRC={:.4f} | "
            "Union P={:.4f} R={:.4f} F1={:.4f} AUPRC={:.4f} | "
            "MeanF1={:.4f}".format(
                region,
                class1_precision,
                class1_recall,
                class1_f1,
                class1_auprc,
                union_precision,
                union_recall,
                union_f1,
                union_auprc,
                mean_f1,
            )
        )

    with out_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print(
        "\nSaved metrics:",
        out_csv,
    )

    print(
        "Saved thresholds:",
        threshold_csv,
    )


if __name__ == "__main__":
    main()
