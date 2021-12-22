_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/chase_db1.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ]),
    test_cfg=dict(crop_size=(128, 128), stride=(85, 85)))
evaluation = dict(metric='mDice')
