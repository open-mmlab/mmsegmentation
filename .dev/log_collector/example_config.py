work_dir = '../mmsegmentation-master/work_dirs'
metric = 'mIoU'
# Don't specify the log_items and ignore_keywords at the same time
log_items = [
    'segformer_mit-b5_512x512_160k_ade20k_cnn_lr_with_warmup',
    'segformer_mit-b5_512x512_160k_ade20k_cnn_no_wramup_lr',
    'segformer_mit-b5_512x512_160k_ade20k_mit_trans_lr',
    'segformer_mit-b5_512x512_160k_ade20k_swin_trans_lr'
]
# or
ignore_keywords = ['segformer']

# should not include metric
other_info_keys = ['other_key']
markdown_file = 'markdowns/lr_in_trans.json.md'
json_file = 'trans_in_cnn.json'
