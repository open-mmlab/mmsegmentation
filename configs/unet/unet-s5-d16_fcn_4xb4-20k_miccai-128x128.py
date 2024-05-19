_base_ = [
    "../_base_/models/fcn_unet_s5-d16.py",
    "../_base_/datasets/miccai.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_20k.py",
]
crop_size = (128, 128)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(128, 128), stride=(85, 85)),
)

# SwanLab
custom_imports = dict(
    imports=["swanlab.integration.mmengine"], allow_failed_imports=False
)

vis_backends = [
    dict(
        type="SwanlabVisBackend",
        save_dir="runs/swanlab",
        init_kwargs={
            "project": "MICCAI",
            "experiment_name": "unet baseline",
            "workspace": "SwanLab",
        },
    ),
]

visualizer = dict(
    type="Visualizer",
    vis_backends=vis_backends,
)
