_base_ = '../deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k.py'

custom_hooks = [
    dict(
        type='RFSearchHook',
        mode='search',
        rfstructure_file=None,
        verbose=True,
        by_epoch=False,
        config=dict(
            search=dict(
                step=0,
                max_step=64001,
                search_interval=8000,
                exp_rate=0.15,
                init_alphas=0.01,
                mmin=1,
                mmax=64,
                num_branches=3,
                skip_layer=[
                    'stem', 'conv1', 'layer1', 'layer2', 'layer3',
                    'auxiliary_head'
                ])))
]

find_unused_parameters = True
