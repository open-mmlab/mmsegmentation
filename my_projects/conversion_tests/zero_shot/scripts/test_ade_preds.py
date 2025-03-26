from mmengine import Config
from mmseg.apis import init_model, inference_model
from mmseg.visualization import SegLocalVisualizer
from mmseg.utils import ade_classes, ade_palette
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
cfg_path = "my_projects/conversion_tests/zero_shot/models/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py"
config = Config.fromfile(cfg_path)

checkpoint_path = "my_projects/conversion_tests/zero_shot/models/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512/weights.pth"

model = init_model(
    config=config,
    checkpoint=checkpoint_path
)
img_1_pth = "/media/ids/Ubuntu files/data/HOTS_v1/SemanticSegmentation/img_dir/test/mix_2_top_raw_6.png"
img_2_pth = "/media/ids/Ubuntu files/data/HOTS_v1/SemanticSegmentation/img_dir/test/office_5_top_raw_2.png"

img_1 = Image.open(img_1_pth)
img_2 = Image.open(img_2_pth)
vis = SegLocalVisualizer(
    classes=ade_classes(),
    palette=ade_palette()
)
out1 = inference_model(
    model=model,
    img=img_1_pth
)

# out2 = inference_model(
#     model=model,
#     img=img_2
# )

mask1 = vis._draw_sem_seg(
    image=np.asarray(img_1.copy()),
    sem_seg=out1.pred_sem_seg,
    classes=ade_classes(),
    palette=ade_palette(),
    with_labels=True
)

plt.imshow(mask1)
plt.show()