git clone https://VIncentmuyi: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@github.com/VIncentmuyi/Floodnet.git

/解决python不搜索本文件夹包的问题
ModuleNotFoundError: No module named 'mmseg'
export PYTHONPATH=.:$PYTHONPATH

ps -ef | grep python
pkill -9 python
git fetch origin
git reset --hard origin/main
git clean -fd  # 删除未跟踪的文件

python tools/train.py ./configs/deeplabv3plus/Deeplabv3+UAVflood.py --work-dir work_dirs/SAR/Deeplabv3+
python tools/test_full_metrics.py ./configs/deeplabv3plus/Deeplabv3+UAVflood.py work_dirs/SAR/Deeplabv3+/best_mIoU_epoch_100.pth  --work-dir ./Result/SAR/Deeplabv3+ --show-dir ./Result/SAR/Deeplabv3+/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/deeplabv3plus/Deeplabv3+UAVflood.py \
    work_dirs/UAVflood/Deeplabv3+/best_val_mIoU_iter_20000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/deeplabv3plus/Deeplabv3+UAVflood.py \
    --shape 8 256 256

python tools/train.py ./configs/segformer/segformer_mit-b0_8xb1-160k_UAVflood-256x256.py --work-dir work_dirs/SAR/segformer
python tools/test_full_metrics.py ./configs/segformer/segformer_mit-b0_8xb1-160k_UAVflood-256x256.py  work_dirs/SAR/segformer/best_mIoU_epoch_100.pth --work-dir ./Result/SAR/segformer/ --show-dir ./Result/SAR/segformer/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/segformer/segformer_mit-b0_8xb1-160k_UAVflood-256x256.py\
    work_dirs/UAVflood/segformer/best_val_mIoU_iter_40000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/segformer/segformer_mit-b0_8xb1-160k_UAVflood-256x256.py \
    --shape 5 256 256

python tools/train.py ./configs/unet/Unet-Uavflood.py --work-dir work_dirs/SARflood/unet
python tools/test_full_metrics.py ./configs/unet/Unet-Uavflood.py  work_dirs/SAR/unet/best_mIoU_epoch_90.pth --work-dir ./Result/SAR/unet/  --show-dir ./Result/SAR/unet/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/unet/Unet-Uavflood.py \
    work_dirs/UAVflood/unet/best_val_mIoU_iter_40000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/unet/Unet-Uavflood.py\
    --shape 8 256 256

python tools/train.py ./configs/mae/mae-base-Uavflood.py --work-dir work_dirs/UAVflood/mae
python tools/test_full_metrics.py ./configs/mae/mae-base-Uavflood.py   work_dirs/UAVflood/mae/best_val_mIoU_iter_20000.pth --work-dir ./Result/UAV/mae/ --show-dir ./Result/UAV/mae/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/mae/mae-base-Uavflood.py\
    work_dirs/UAVflood/mae/best_val_mIoU_iter_28000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/mae/mae-base-Uavflood.py\
    --shape 5 256 256

python tools/train.py ./configs/vit/vit-Uavflood.py --work-dir work_dirs/SAR/vit
python tools/test_full_metrics.py ./configs/vit/vit-Uavflood.py  work_dirs/SAR/vit/best_mIoU_epoch_100.pth --work-dir ./Result/SAR/vit/  --show-dir ./Result/SAR/vit/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/vit/vit-Uavflood.py\
    work_dirs/UAVflood/vit/best_val_mIoU_iter_36000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/vit/vit-Uavflood.py\
    --shape 3 256 256

python tools/train.py ./configs/beit/beit-Uavflood.py --work-dir work_dirs/SARflood/beit
python tools/test_full_metrics.py ./configs/beit/beit-Uavflood.py  work_dirs/UAVflood/beit/best_val_mIoU_iter_16000.pth --work-dir ./Result/UAV/beit/  --show-dir ./Result/UAV/beit/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/beit/beit-Uavflood.py\
    work_dirs/UAVflood/beit/best_val_mIoU_iter_40000.pth\
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/beit/beit-Uavflood.py\
    --shape 3 256 256

python tools/train.py ./configs/convnext/convnext-base-uavflood.py --work-dir work_dirs/SAR/convnext
python tools/test_full_metrics.py ./configs/convnext/convnext-base-uavflood.py  work_dirs/SAR/convnext/best_mIoU_epoch_100.pth --work-dir ./Result/SAR/convnext/  --show-dir ./Result/SAR/convnext/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/convnext/convnext-base-uavflood.py\
    work_dirs/GFflood/convnext/best_val_mIoU_iter_20000.pth \
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/convnext/convnext-base-uavflood.py\
    --shape 5 256 256

python tools/train.py ./configs/swin/Swin-uavflood-256x256.py --work-dir work_dirs/SAR/Swin --cfg-options seed=42
python tools/test_full_metrics.py ./configs/swin/Swin-uavflood-256x256.py  work_dirs/GFflood/Swin/best_val_mIoU_iter_16000.pth --work-dir ./Result/GF/swin/  --show-dir ./Result/GF/swin/vis --cfg-options visualizer.alpha=1.0
python tools/analysis_tools/benchmark.py \
    ./configs/swin/Swin-uavflood-256x256.py\
    work_dirs/GFflood/Swin/best_val_mIoU_iter_16000.pth\
    --repeat-times 3
python tools/analysis_tools/get_flops.py \
    ./configs/swin/Swin-uavflood-256x256.py\
    --shape 5 256 256

python tools/train.py ./configs/floodnet/multimodal_floodnet_sar_boost_swinbase_moe_config.py --work-dir work_dirs/floodnet/SwinmoeB/655 --cfg-options seed=42
python tools/test_full_metrics.py ./configs/floodnet/multimodal_floodnet_sar_boost_swin_moe_config.py  work_dirs/floodnet/SwinmoeB/best_mIoU_epoch_100.pth --work-dir ./Result/Floodnet/SAR/  --show-dir ./Result/Floodnet/SAR/vis --cfg-options visualizer.alpha=1.0

python tools/test_full_metrics.py \
  configs/floodnet/multimodal_floodnet_sar_boost_swinbase_moe_config.py \
  work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth \
  --cfg-options test_dataloader.dataset.filter_modality=sar \
  --work-dir ./Result/Floodnet/SAR/  \
  --show-dir ./Result/Floodnet/SAR/vis --cfg-options visualizer.alpha=1.0


python tools/train.py \
  configs/floodnet/multimodal_floodnet_sar_only_swinbase_moe_config.py \
  --work-dir work_dirs/floodnet/SwinmoeB_sar_only \
  --cfg-options seed=42

python tools/test_full_metrics.py \
    configs/floodnet/multimodal_floodnet_sar_only_swinbase_moe_config.py \
    work_dirs/floodnet/SwinmoeB_sar_only/best_mIoU_epoch_100.pth \


 python tools/train.py configs/floodnet/continue_train_150ep.py --work-dir work_dirs/floodnet/SwinmoeB/655 --resume --cfg-options load_from="work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth"

 python tools/analysis_tools/visualize_expert_routing.py \
    configs/floodnet/multimodal_floodnet_sar_boost_swinbase_moe_config.py \
    work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth \
    --output-dir work_dirs/figures/expert_routing \
    --num-samples 50

python tools/train.py configs/floodnet/finetune_single_modal.py \
    --work-dir work_dirs/generalization/LY-train-station/ \
    --cfg-options \
        train_dataloader.dataset.data_root="data/LY-train-station/" \
        val_dataloader.dataset.data_root="data/LY-train-station/" \
        test_dataloader.dataset.data_root="data/LY-train-station/"

python tools/test.py \
    configs/floodnet/finetune_single_modal.py \
    work_dirs/generalization/LY-train-station/best_mIoU_epoch_50.pth \
    --work-dir work_dirs/generalization/LY-train-station/test_results \
    --cfg-options \
        test_dataloader.dataset.data_root="data/LY-train-station/" \
        "test_evaluator.iou_metrics=['mIoU','mDice','mFscore']" \
    --show-dir work_dirs/generalization/LY-train-station/test_results/vis \
    --out work_dirs/generalization/LY-train-station/test_results/predictions

python tools/predict_large_tif.py \
    configs/floodnet/finetune_single_modal.py \
    work_dirs/generalization/LY-train-station/best_mIoU_epoch_50.pth \
    --input data/luoyuan/result.tif \
    --output data/luoyuan/prediction.tif \
    --tile-size 512 \
    --overlap 64 \
    --modal rgb \
    --bands 0 1 2 \
    --batch-size 16