from mmseg.apis import MMSegInferencer
# hack to load custom models
import mmseg.models.backbones.mobile_sam_vit
import mmseg.models.backbones.sam_vit
import mmseg.engine.hooks.force_test_loop_hook
import mmseg.engine.hooks.best_model_testing_hook

from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from PIL import Image
import pathlib
import time

def main(config):
    inference = MMSegInferencer(
        model=config.config,
        weights=config.checkpoint,
        classes=("full", "empty",),
        palette=([0,0,0], [0,255,0],),
    )
    images : Path = config.images_dir
    output_dir: Path = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if config.mask_only:
        for p in images.glob("**/*.jpg"):
            print("---", p)
            result = inference(str(p))
            
            mask = result['predictions'].astype(np.uint8) * 255
            mask_image = Image.fromarray(mask)
            output_path = pathlib.Path(output_dir) / p.name
            
            mask_image.save(output_path)

    else:
        for p in images.glob("**/*.jpg"):
            print(p)
            inference(
                str(p),
                out_dir=str(output_dir)
            )
    print(f"Inference time: {round(time.time() - start_time, 2)} s.")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("config")
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--mask_only",
        action="store_true",
        help="save masks only")
    config = parser.parse_args()
    main(config)


