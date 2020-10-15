python tools/convert_datasets/chase_db1.py /mnt/lustre/share/yamengxi/all_data/CHASE_DB1/CHASEDB1.zip \
--tmp_dir=tmp

python tools/convert_datasets/drive.py /mnt/lustre/share/yamengxi/all_data/DRIVE/training.zip /mnt/lustre/share/yamengxi/all_data/DRIVE/test.zip \
--tmp_dir=tmp

python tools/convert_datasets/hrf.py \
/mnt/lustre/share/yamengxi/all_data/HRF/healthy.zip \
/mnt/lustre/share/yamengxi/all_data/HRF/healthy_manualsegm.zip \
/mnt/lustre/share/yamengxi/all_data/HRF/glaucoma.zip \
/mnt/lustre/share/yamengxi/all_data/HRF/glaucoma_manualsegm.zip \
/mnt/lustre/share/yamengxi/all_data/HRF/diabetic_retinopathy.zip \
/mnt/lustre/share/yamengxi/all_data/HRF/diabetic_retinopathy_manualsegm.zip \
--tmp_dir=tmp

python tools/convert_datasets/stare.py \
/mnt/lustre/share/yamengxi/all_data/STARE/stare-images.tar \
/mnt/lustre/share/yamengxi/all_data/STARE/labels-ah.tar \
/mnt/lustre/share/yamengxi/all_data/STARE/labels-vk.tar \
--tmp_dir=tmp
