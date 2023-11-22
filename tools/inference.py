from mmseg.apis import MMSegInferencer
# hack to load custom models
import mmseg.models.backbones.mobile_sam_vit
import mmseg.models.backbones.sam_vit
import mmseg.engine.hooks.logger_hook_force_test
import mmseg.engine.hooks.best_model_testing_hook

from pathlib import Path
from argparse import ArgumentParser

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

    for p in images.glob("**/*.jpg"):
        print(p)
        inference(
            str(p),
            out_dir=str(output_dir)
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("config")
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    config = parser.parse_args()
    main(config)


