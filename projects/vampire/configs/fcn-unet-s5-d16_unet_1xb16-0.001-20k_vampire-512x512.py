_base_ = [
    'mmseg::_base_/models/fcn_unet_s5-d16.py', './vampire_512x512.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_20k.py'
]
custom_imports = dict(imports='datasets.vampire_dataset')
img_scale = (512, 512)
data_preprocessor = dict(size=img_scale)
optimizer = dict(lr=0.001)
optim_wrapper = dict(optimizer=optimizer)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(size=img_scale),
    pretrained=None,
    decode_head=dict(num_classes=2),
    auxiliary_head=None,
    test_cfg=dict(mode='whole', _delete_=True))
vis_backends = None
visualizer = dict(vis_backends=vis_backends)
work_dir = 'projects/vampire/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.001-20k_vampire-512x512/'  # noqa
