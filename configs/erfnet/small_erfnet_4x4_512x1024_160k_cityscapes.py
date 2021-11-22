_base_ = './erfnet_4x4_512x1024_160k_cityscapes.py'
model = dict(decode_head=dict(num_convs=1, concat_input=False))
