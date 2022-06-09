work_dir = '../../work_dirs'
metric = 'mIoU'

# specify the log files we would like to collect in `log_items`
log_items = [
    'segformer_mit-b5_512x512_160k_ade20k_cnn_lr_with_warmup',
    'segformer_mit-b5_512x512_160k_ade20k_cnn_no_warmup_lr',
    'segformer_mit-b5_512x512_160k_ade20k_mit_trans_lr',
    'segformer_mit-b5_512x512_160k_ade20k_swin_trans_lr'
]
# or specify ignore_keywords, then the folders whose name contain
# `'segformer'` won't be collected
# ignore_keywords = ['segformer']

# should not include metric
other_info_keys = ['mAcc']
markdown_file = 'markdowns/lr_in_trans.json.md'
json_file = 'jsons/trans_in_cnn.json'
