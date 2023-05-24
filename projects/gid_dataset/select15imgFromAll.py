import shutil

# select 15 images from GID dataset

img_list = [
    'GF2_PMS1__L1A0000647767-MSS1.tif', 'GF2_PMS1__L1A0001064454-MSS1.tif',
    'GF2_PMS1__L1A0001348919-MSS1.tif', 'GF2_PMS1__L1A0001680851-MSS1.tif',
    'GF2_PMS1__L1A0001680853-MSS1.tif', 'GF2_PMS1__L1A0001680857-MSS1.tif',
    'GF2_PMS1__L1A0001757429-MSS1.tif', 'GF2_PMS2__L1A0000607681-MSS2.tif',
    'GF2_PMS2__L1A0000635115-MSS2.tif', 'GF2_PMS2__L1A0000658637-MSS2.tif',
    'GF2_PMS2__L1A0001206072-MSS2.tif', 'GF2_PMS2__L1A0001471436-MSS2.tif',
    'GF2_PMS2__L1A0001642620-MSS2.tif', 'GF2_PMS2__L1A0001787089-MSS2.tif',
    'GF2_PMS2__L1A0001838560-MSS2.tif'
]

labels_list = [
    'GF2_PMS1__L1A0000647767-MSS1_label.tif',
    'GF2_PMS1__L1A0001064454-MSS1_label.tif',
    'GF2_PMS1__L1A0001348919-MSS1_label.tif',
    'GF2_PMS1__L1A0001680851-MSS1_label.tif',
    'GF2_PMS1__L1A0001680853-MSS1_label.tif',
    'GF2_PMS1__L1A0001680857-MSS1_label.tif',
    'GF2_PMS1__L1A0001757429-MSS1_label.tif',
    'GF2_PMS2__L1A0000607681-MSS2_label.tif',
    'GF2_PMS2__L1A0000635115-MSS2_label.tif',
    'GF2_PMS2__L1A0000658637-MSS2_label.tif',
    'GF2_PMS2__L1A0001206072-MSS2_label.tif',
    'GF2_PMS2__L1A0001471436-MSS2_label.tif',
    'GF2_PMS2__L1A0001642620-MSS2_label.tif',
    'GF2_PMS2__L1A0001787089-MSS2_label.tif',
    'GF2_PMS2__L1A0001838560-MSS2_label.tif'
]

img_root_path = r'\image_RGB'
label_root_path = r'\label_5classes'

dest_img_dir = r'D:\ATL\AI_work\Datasets\GID\15\images'
dest_label_dir = r'D:\ATL\AI_work\Datasets\GID\15\labels'

# 把img_list的文件复制到 desr_dir
for img in img_list:
    shutil.copy(img_root_path + '\\' + img, dest_img_dir)

for label in labels_list:
    shutil.copy(label_root_path + '\\' + label, dest_label_dir)
