_base_ = './pascal_voc12.py'
# dataset settings
train_dataloader = dict(
    dataset=dict(
        ann_dir=['SegmentationClass', 'SegmentationClassAug'],
        ann_file=[
            'ImageSets/Segmentation/train.txt',
            'ImageSets/Segmentation/aug.txt'
        ]))
