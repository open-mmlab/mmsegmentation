from __future__ import annotations

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RemapThreeClassToBinary(BaseTransform):
    """
    Existing 3-class mask:
        0   = background
        1   = confirmed_filament
        2   = candidate_filament
        255 = ignore

    background_vs_confirmed:
        0 -> 0
        1 -> 1
        2 -> 255
        255 -> 255

    background_vs_candidate:
        0 -> 0
        2 -> 1
        1 -> 255
        255 -> 255
    """

    def __init__(self, mode: str):
        allowed = {
            "background_vs_confirmed",
            "background_vs_candidate",
        }
        if mode not in allowed:
            raise ValueError(
                f"Unsupported mode={mode!r}. "
                f"Expected one of {sorted(allowed)}"
            )
        self.mode = mode

    def transform(self, results: dict) -> dict:
        src = np.asarray(results["gt_seg_map"])
        dst = np.full(src.shape, 255, dtype=np.uint8)

        dst[src == 0] = 0

        if self.mode == "background_vs_confirmed":
            dst[src == 1] = 1
        else:
            dst[src == 2] = 1

        results["gt_seg_map"] = dst
        return results
