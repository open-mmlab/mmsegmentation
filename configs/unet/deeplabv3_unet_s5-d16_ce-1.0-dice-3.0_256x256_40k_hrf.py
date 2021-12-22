_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', '../_base_/datasets/hrf.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ]),
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))
evaluation = dict(metric='mDice')
