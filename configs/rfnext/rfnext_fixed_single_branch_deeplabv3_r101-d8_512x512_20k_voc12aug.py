_base_ = '../deeplabv3/deeplabv3_r101-d8_512x512_20k_voc12aug.py'

custom_hooks = [
    dict(
        type='RFSearchHook',
        mode='fixed_single_branch',
        rfstructure_file='./configs/rfnext/search_log/deeplabv3_r101-d8_512x512_20k_voc12aug/local_search_config_step16000.json',
        verbose=True,
        by_epoch=False,
        config=dict(
            search=dict(
                step=0,
                max_step=16001,
                search_interval=2000,
                exp_rate=0.15,
                init_alphas=0.01,
                mmin=1,
                mmax=64,
                num_branches=3,
                skip_layer=['stem', 'conv1', 'layer1', 'layer2', 'layer3', 'auxiliary_head'])))
]