"""
Continue training Full Model for 50 more epochs (101-150).
"""

_base_ = ['./multimodal_floodnet_sar_boost_swinbase_moe_config.py']

# Extend to 150 epochs
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=150,
    val_interval=10)

# Extend PolyLR end to 150
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=5),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=5,
        end=150,
        by_epoch=True),
]
