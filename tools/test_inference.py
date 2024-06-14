import os
import os.path as osp
from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

proj_path = "work_dirs/convnext-tiny_upernet_1xb2-500_hots-v1-512x512"
cfg_name = [file for file in os.listdir(proj_path) if '.py' in file][0]
checkpoint_name = [file for file in os.listdir(proj_path) if '.pth' in file][0]
cfg_path = os.path.join(proj_path, cfg_name)
checkpoint_path = os.path.join(proj_path, checkpoint_name)

print(f"cfg_path: {cfg_path}\ncheckpoint_path: {checkpoint_path}")

cfg = Config.fromfile(filename=cfg_path)
inferencer = init_model(config=cfg, checkpoint=checkpoint_path, device='cuda:0')

img = "/media/ids/Ubuntu files/data/HOTS_v1/SemanticSegmentation/img_dir/test/kitchen_10_top_raw_3.png"
result = inference_model(inferencer, img=img)
print(result)
show_result_pyplot(inferencer, img, result, show=True)
exit()
dataloader_cfg = cfg["test_dataloader"]
test_loader = Runner.build_dataloader(dataloader_cfg)
dataset = test_loader.dataset
for item in dataset:
    # print(item)
    print('#' * 50)
    inputs = item['inputs']
    gt_sem_seg = item['data_samples'].gt_sem_seg.data
    res = inferencer(inputs)
    # print(f"inputs: \n {inputs.shape}\n GT:\n{gt_sem_seg.shape}")
    # print(f"res:\n {res}") 
print(f"{'#' * 80}\n{'#' * 30} TRAIN {'#' * 30}\n{'#' * 80}")
dataloader_cfg = cfg["train_dataloader"]
train_loader = Runner.build_dataloader(dataloader_cfg)
dataset = train_loader.dataset
for item in dataset:
    # print(item)
    print('#' * 50)
    inputs = item['inputs']
    gt_sem_seg = item['data_samples'].gt_sem_seg.data
    print(f"inputs: \n {inputs.shape}\n GT:\n{gt_sem_seg.shape}") 
# res = inference_model(model=inferencer, )

#  runner = Runner.from_cfg(cfg=cfg)
#             test_loader = runner.test_dataloader
#             # torch.set_printoptions(profile="full")
#             import numpy as np
#             for item in test_loader:
#                 print(f"item: \n{item['inputs']}")
#                 for data in list(item['data_samples'][0].gt_sem_seg.values()):
#                     print(np.unique(data[0]))
#                     print("in")
#                     print(np.unique(item['inputs'][0]))
#                 # print(f"item:")
#                 # for key, val in item.items():
#                 #     print(f"{key} : {val}")
#             exit()
#             runner.test()