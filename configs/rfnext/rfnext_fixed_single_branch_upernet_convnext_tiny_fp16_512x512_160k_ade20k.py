_base_ = '../convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k.py'

custom_hooks = [
    dict(
        type='RFSearchHook',
        mode='fixed_single_branch',
        rfstructure_file='./configs/rfnext/search_log/upernet_convnext_tiny_fp16_512x512_160k_ade20k/local_search_config_step64000.json',
        verbose=True,
        by_epoch=False,
        config=dict(
            search=dict(
                step=0,
                max_step=64001,
                search_interval=8000,
                exp_rate=0.5,
                init_alphas=0.01,
                mmin=1,
                mmax=24,
                num_branches=3,
                skip_layer=['stages.0', 'stages.1', 'stages.2', 'auxiliary_head'])))
]
