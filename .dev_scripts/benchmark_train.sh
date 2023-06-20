PARTITION=$1

echo 'configs/hrnet/fcn_hr18s_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION fcn_hr18s_512x512_160k_ade20k configs/hrnet/fcn_hr18s_512x512_160k_ade20k.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24727 --work-dir work_dirs/hrnet/fcn_hr18s_512x512_160k_ade20k >/dev/null &
echo 'configs/hrnet/fcn_hr18s_512x1024_160k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION fcn_hr18s_512x1024_160k_cityscapes configs/hrnet/fcn_hr18s_512x1024_160k_cityscapes.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24728 --work-dir work_dirs/hrnet/fcn_hr18s_512x1024_160k_cityscapes >/dev/null &
echo 'configs/hrnet/fcn_hr48_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION fcn_hr48_512x512_160k_ade20k configs/hrnet/fcn_hr48_512x512_160k_ade20k.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24729 --work-dir work_dirs/hrnet/fcn_hr48_512x512_160k_ade20k >/dev/null &
echo 'configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION fcn_hr48_512x1024_160k_cityscapes configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24730 --work-dir work_dirs/hrnet/fcn_hr48_512x1024_160k_cityscapes >/dev/null &
echo 'configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION pspnet_r50-d8_512x1024_80k_cityscapes configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24731 --work-dir work_dirs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes >/dev/null &
echo 'configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION pspnet_r101-d8_512x1024_80k_cityscapes configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24732 --work-dir work_dirs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes >/dev/null &
echo 'configs/pspnet/pspnet_r101-d8_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION pspnet_r101-d8_512x512_160k_ade20k configs/pspnet/pspnet_r101-d8_512x512_160k_ade20k.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24733 --work-dir work_dirs/pspnet/pspnet_r101-d8_512x512_160k_ade20k >/dev/null &
echo 'configs/pspnet/pspnet_r50-d8_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION pspnet_r50-d8_512x512_160k_ade20k configs/pspnet/pspnet_r50-d8_512x512_160k_ade20k.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24734 --work-dir work_dirs/pspnet/pspnet_r50-d8_512x512_160k_ade20k >/dev/null &
echo 'configs/resnest/pspnet_s101-d8_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION pspnet_s101-d8_512x512_160k_ade20k configs/resnest/pspnet_s101-d8_512x512_160k_ade20k.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24735 --work-dir work_dirs/resnest/pspnet_s101-d8_512x512_160k_ade20k >/dev/null &
echo 'configs/resnest/pspnet_s101-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION pspnet_s101-d8_512x1024_80k_cityscapes configs/resnest/pspnet_s101-d8_512x1024_80k_cityscapes.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24736 --work-dir work_dirs/resnest/pspnet_s101-d8_512x1024_80k_cityscapes >/dev/null &
echo 'configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION fast_scnn_lr0.12_8x4_160k_cityscapes configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24737 --work-dir work_dirs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes >/dev/null &
echo 'configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION deeplabv3plus_r101-d8_769x769_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24738 --work-dir work_dirs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes >/dev/null &
echo 'configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION deeplabv3plus_r101-d8_512x1024_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24739 --work-dir work_dirs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes >/dev/null &
echo 'configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION deeplabv3plus_r50-d8_512x1024_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24740 --work-dir work_dirs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes >/dev/null &
echo 'configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION deeplabv3plus_r50-d8_769x769_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24741 --work-dir work_dirs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes >/dev/null &
echo 'configs/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION upernet_vit-b16_ln_mln_512x512_160k_ade20k configs/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24742 --work-dir work_dirs/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k >/dev/null &
echo 'configs/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION upernet_deit-s16_ln_mln_512x512_160k_ade20k configs/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24743 --work-dir work_dirs/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k >/dev/null &
echo 'configs/deeplabv3plus/deeplabv3plus_r101-d8_fp16_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes configs/deeplabv3plus/deeplabv3plus_r101-d8_fp16_512x1024_80k_cityscapes.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24744 --work-dir work_dirs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes >/dev/null &
echo 'configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 ./tools/slurm_train.sh $PARTITION upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py --cfg-options checkpoint_config.max_keep_ckpts=1 dist_params.port=24745 --work-dir work_dirs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K >/dev/null &
