_base_ = [
    'mmseg::_base_/models/fcn_unet_s5-d16.py', './segpc2021_512x512.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_20k.py'
]
custom_imports = dict(imports='projects.segpc2021.datasets.segpc2021_dataset')
img_scale = (512, 512)
data_preprocessor = dict(size=img_scale)
optimizer = dict(lr=0.01)
optim_wrapper = dict(optimizer=optimizer)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    decode_head=dict(num_classes=3),
    auxiliary_head=None,
    test_cfg=dict(mode='whole', _delete_=True))
vis_backends = None
visualizer = dict(vis_backends=vis_backends)
work_dir = 'projects/segpc2021/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.01-20k_segpc2021-512x512/'  # noqa
