GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf pr_test configs/hrnet/fcn_hr48_480x480_40k_pascal_context.py --seed=0
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf v3+_pc configs/deeplabv3plus/deeplabv3plus_r101-d8_480x480_40k_pascal_context.py --seed=0
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf v3_pc configs/deeplabv3/deeplabv3_r101-d8_480x480_40k_pascal_context.py --seed=0 --options model.backbone.with_cp=True
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf psp_pc configs/pspnet/pspnet_r101-d8_480x480_40k_pascal_context.py --seed=0
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf fcn_pc configs/fcn/fcn_r101-d8_480x480_40k_pascal_context.py --seed=0


GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf1 pr_test configs/hrnet/fcn_hr48_480x480_80k_pascal_context.py --seed=0
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf1 v3+_pc configs/deeplabv3plus/deeplabv3plus_r101-d8_480x480_80k_pascal_context.py --seed=0
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf1 v3_pc configs/deeplabv3/deeplabv3_r101-d8_480x480_80k_pascal_context.py --seed=0 --options model.backbone.with_cp=True
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf1 psp_pc configs/pspnet/pspnet_r101-d8_480x480_80k_pascal_context.py --seed=0
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf1 fcn_pc configs/fcn/fcn_r101-d8_480x480_80k_pascal_context.py --seed=0

# # test
# GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_test.sh Test test configs/hrnet/fcn_hr48_480x480_80k_pascal_context.py /mnt/lustre/yamengxi/noob/mmsegmentation/bbb.pth --tmpdir tmpdir --eval mIoU
