_base_ = './uniformer_upernet_global_base_1xb2-160k_ade20k-512x512.py'
model = dict(backbone=dict(layers=[3, 4, 8, 3], drop_path_rate=0.25))
# load_from = '/root/mmsegmentation/upernet_global_small.pth'
