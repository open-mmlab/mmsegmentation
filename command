git clone https://VIncentmuyi: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@github.com/VIncentmuyi/Floodnet.git

/解决python不搜索本文件夹包的问题
ModuleNotFoundError: No module named 'mmseg'
export PYTHONPATH=.:$PYTHONPATH

ps -ef | grep python
pkill -9 python


python tools/train.py ./configs/deeplabv3plus/Deeplabv3+UAVflood.py --work-dir work_dirs/SAR/Deeplabv3+
python tools/test.py ./configs/deeplabv3plus/Deeplabv3+UAVflood.py work_dirs/SAR/Deeplabv3+/best_mIoU_epoch_100.pth  --work-dir ./Result/SAR/Deeplabv3+ --show-dir ./Result/SAR/Deeplabv3+/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/deeplabv3plus/Deeplabv3+UAVflood.py \
    work_dirs/UAVflood/Deeplabv3+/best_val_mIoU_iter_20000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/deeplabv3plus/Deeplabv3+UAVflood.py \
    --shape 8 256 256

python tools/train.py ./configs/segformer/segformer_mit-b0_8xb1-160k_UAVflood-256x256.py --work-dir work_dirs/SAR/segformer
python tools/test.py ./configs/segformer/segformer_mit-b0_8xb1-160k_UAVflood-256x256.py  work_dirs/UAVflood/segformer/best_val_mIoU_iter_20000.pth --work-dir ./Result/UAV/segformer/ --show-dir ./Result/UAV/segformer/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/segformer/segformer_mit-b0_8xb1-160k_UAVflood-256x256.py\
    work_dirs/UAVflood/segformer/best_val_mIoU_iter_40000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/segformer/segformer_mit-b0_8xb1-160k_UAVflood-256x256.py \
    --shape 5 256 256

python tools/train.py ./configs/unet/Unet-Uavflood.py --work-dir work_dirs/SARflood/unet
python tools/test.py ./configs/unet/Unet-Uavflood.py  work_dirs/SARflood/unet/best_val_mIoU_iter_20000.pth --work-dir ./Result/SAR/unet/  --show-dir ./Result/SAR/unet/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/unet/Unet-Uavflood.py \
    work_dirs/UAVflood/unet/best_val_mIoU_iter_40000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/unet/Unet-Uavflood.py\
    --shape 8 256 256

python tools/train.py ./configs/mae/mae-base-Uavflood.py --work-dir work_dirs/UAVflood/mae
python tools/test.py ./configs/mae/mae-base-Uavflood.py   work_dirs/UAVflood/mae/best_val_mIoU_iter_20000.pth --work-dir ./Result/UAV/mae/ --show-dir ./Result/UAV/mae/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/mae/mae-base-Uavflood.py\
    work_dirs/UAVflood/mae/best_val_mIoU_iter_28000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/mae/mae-base-Uavflood.py\
    --shape 5 256 256

python tools/train.py ./configs/vit/vit-Uavflood.py --work-dir work_dirs/SARflood/vit
python tools/test.py ./configs/vit/vit-Uavflood.py  work_dirs/SARflood/vit/best_val_mIoU_iter_16000.pth --work-dir ./Result/SAR/vit/  --show-dir ./Result/SAR/vit/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/vit/vit-Uavflood.py\
    work_dirs/UAVflood/vit/best_val_mIoU_iter_36000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/vit/vit-Uavflood.py\
    --shape 3 256 256

python tools/train.py ./configs/beit/beit-Uavflood.py --work-dir work_dirs/SARflood/beit
python tools/test.py ./configs/beit/beit-Uavflood.py  work_dirs/UAVflood/beit/best_val_mIoU_iter_16000.pth --work-dir ./Result/UAV/beit/  --show-dir ./Result/UAV/beit/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/beit/beit-Uavflood.py\
    work_dirs/UAVflood/beit/best_val_mIoU_iter_40000.pth\
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/beit/beit-Uavflood.py\
    --shape 3 256 256

python tools/train.py ./configs/convnext/convnext-base-uavflood.py --work-dir work_dirs/SAR/convnext
python tools/test.py ./configs/convnext/convnext-base-uavflood.py  work_dirs/GFflood/convnext/best_val_mIoU_iter_20000.pth --work-dir ./Result/GF/convnext/  --show-dir ./Result/GF/convnext/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/convnext/convnext-base-uavflood.py\
    work_dirs/GFflood/convnext/best_val_mIoU_iter_20000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/convnext/convnext-base-uavflood.py\
    --shape 5 256 256

python tools/train.py ./configs/swin/Swin-uavflood-256x256.py --work-dir work_dirs/GFflood/Swin --cfg-options seed=42
python tools/test.py ./configs/swin/Swin-uavflood-256x256.py  work_dirs/GFflood/Swin/best_val_mIoU_iter_16000.pth --work-dir ./Result/GF/swin/  --show-dir ./Result/UAV/swin/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/swin/Swin-uavflood-256x256.py\
    work_dirs/GFflood/Swin/best_val_mIoU_iter_16000.pth\
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/swin/Swin-uavflood-256x256.py\
    --shape 5 256 256

