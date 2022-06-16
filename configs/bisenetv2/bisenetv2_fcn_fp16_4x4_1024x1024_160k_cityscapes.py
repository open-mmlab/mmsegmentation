_base_ = './bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py'
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=optimizer,
    loss_scale=512.)
