_base_ = './segmenter_vit-s_mask_8xb1-20k_HOTS_v1-640x480.py'

model = dict(
    decode_head=dict(
        _delete_=True,
        type='FCNHead',
        in_channels=384,
        channels=384,
        num_convs=0,
        dropout_ratio=0.0,
        concat_input=False,
        num_classes=46,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
