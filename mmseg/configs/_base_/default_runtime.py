from mmengine.runner import LogProcessor
from mmengine.visualization import LocalVisBackend
from mmseg.visualization import SegLocalVisualizer
from mmseg.models.segmentors import SegTTAModel

default_scope = None
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
log_processor = dict(type=LogProcessor, window_size=50, by_epoch=False)

log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type=SegTTAModel)
