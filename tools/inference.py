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


class FPSLogger:
    def __init__(self):
        self.tottime = 0.0
        self.count = 0
        self.last_record = 0.0
        self.last_print = time.time()
        self.interval = 10

    def start_record(self):
        self.last_record = time.time()

    def end_record(self):
        self.tottime += time.time() - self.last_record
        self.count += 1
        self.print_fps()

    def print_fps(self):
        if time.time() - self.last_print > self.interval:
            print(f"Inference running at {self.count / self.tottime:.3f} FPS")
            self.last_print = time.time()


def main(config):
    fps_logger = FPSLogger()
    inference = MMSegInferencer(
        model=config.config,
        weights=config.checkpoint,
        classes=("full", "empty",),
        palette=([0,0,0], [0,255,0],),
    )
    if config.silent:
        inference.show_progress = False

    images : Path = config.images_dir
    output_dir: Path = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if config.mask_only:
        image_paths = list(images.glob("**/*.jpg"))
        num_images = len(image_paths)
        for i, p in enumerate(image_paths, 1):
            try:
                if i % 100 == 0:
                    print(f"|mmseg predict| [{i}/{num_images}]")

                fps_logger.start_record()
                result = inference(str(p))

                mask = result['predictions'].astype(np.uint8) * 255
                mask_image = Image.fromarray(mask)
                output_path = pathlib.Path(output_dir) / p.name
                output_path = output_path.with_suffix(".png")

                mask_image.save(output_path, "PNG")
                fps_logger.end_record()
            
            except Exception as e:
                print(f"Failed with {p}. {e}")

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
    parser.add_argument(
        "--silent",
        action="store_true",
        help="suppress progress bars and verbose output")
    config = parser.parse_args()
    main(config)
