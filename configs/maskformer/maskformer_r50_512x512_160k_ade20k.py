_base_ = ['./maskformer_r50_512x512_80k_ade20k.py']

# model_cfg
runner = dict(type='IterBasedRunner', max_iters=160000)
