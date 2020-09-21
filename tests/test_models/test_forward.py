"""pytest tests/test_forward.py."""
import copy
from os.path import dirname, exists, join
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.utils.parrots_wrapper import SyncBatchNorm, _BatchNorm


def _demo_mm_inputs(input_shape=(2, 3, 8, 16), num_classes=10):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    segs = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'flip_direction': 'horizontal'
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs


def _get_config_directory():
    """Find the predefined segmentor config directory."""
    try:
        # Assume we are running in the source mmsegmentation repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmseg
        repo_dpath = dirname(dirname(dirname(mmseg.__file__)))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_segmentor_cfg(fname):
    """Grab configs necessary to create a segmentor.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg))
    return model, train_cfg, test_cfg


def test_pspnet_forward():
    _test_encoder_decoder_forward(
        'pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')


def test_fcn_forward():
    _test_encoder_decoder_forward('fcn/fcn_r50-d8_512x1024_40k_cityscapes.py')


def test_deeplabv3_forward():
    _test_encoder_decoder_forward(
        'deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py')


def test_deeplabv3plus_forward():
    _test_encoder_decoder_forward(
        'deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py')


def test_gcnet_forward():
    _test_encoder_decoder_forward(
        'gcnet/gcnet_r50-d8_512x1024_40k_cityscapes.py')


def test_ann_forward():
    _test_encoder_decoder_forward('ann/ann_r50-d8_512x1024_40k_cityscapes.py')


def test_ccnet_forward():
    if not torch.cuda.is_available():
        pytest.skip('CCNet requires CUDA')
    _test_encoder_decoder_forward(
        'ccnet/ccnet_r50-d8_512x1024_40k_cityscapes.py')


def test_danet_forward():
    _test_encoder_decoder_forward(
        'danet/danet_r50-d8_512x1024_40k_cityscapes.py')


def test_nonlocal_net_forward():
    _test_encoder_decoder_forward(
        'nonlocal_net/nonlocal_r50-d8_512x1024_40k_cityscapes.py')


def test_upernet_forward():
    _test_encoder_decoder_forward(
        'upernet/upernet_r50_512x1024_40k_cityscapes.py')


def test_hrnet_forward():
    _test_encoder_decoder_forward('hrnet/fcn_hr18s_512x1024_40k_cityscapes.py')


def test_ocrnet_forward():
    _test_encoder_decoder_forward(
        'ocrnet/ocrnet_hr18s_512x1024_40k_cityscapes.py')


def test_psanet_forward():
    _test_encoder_decoder_forward(
        'psanet/psanet_r50-d8_512x1024_40k_cityscapes.py')


def test_encnet_forward():
    _test_encoder_decoder_forward(
        'encnet/encnet_r50-d8_512x1024_40k_cityscapes.py')


def test_sem_fpn_forward():
    _test_encoder_decoder_forward('sem_fpn/fpn_r50_512x1024_80k_cityscapes.py')


def test_point_rend_forward():
    _test_encoder_decoder_forward(
        'point_rend/pointrend_r50_512x1024_80k_cityscapes.py')


def test_mobilenet_v2_forward():
    _test_encoder_decoder_forward(
        'mobilenet_v2/pspnet_m-v2-d8_512x1024_80k_cityscapes.py')


def test_dnlnet_forward():
    _test_encoder_decoder_forward(
        'dnlnet/dnl_r50-d8_512x1024_40k_cityscapes.py')


def test_emanet_forward():
    _test_encoder_decoder_forward(
        'emanet/emanet_r50-d8_512x1024_80k_cityscapes.py')


def get_world_size(process_group):

    return 1


def _check_input_dim(self, inputs):
    pass


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, SyncBatchNorm):
        # to be consistent with SyncBN, we hack dim check function in BN
        module_output = _BatchNorm(module.num_features, module.eps,
                                   module.momentum, module.affine,
                                   module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


@patch('torch.nn.modules.batchnorm._BatchNorm._check_input_dim',
       _check_input_dim)
@patch('torch.distributed.get_world_size', get_world_size)
def _test_encoder_decoder_forward(cfg_file):
    model, train_cfg, test_cfg = _get_segmentor_cfg(cfg_file)
    model['pretrained'] = None
    test_cfg['mode'] = 'whole'

    from mmseg.models import build_segmentor
    segmentor = build_segmentor(model, train_cfg=train_cfg, test_cfg=test_cfg)

    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes
    # batch_size=2 for BatchNorm
    input_shape = (2, 3, 32, 32)
    mm_inputs = _demo_mm_inputs(input_shape, num_classes=num_classes)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_semantic_seg = mm_inputs['gt_semantic_seg']

    # convert to cuda Tensor if applicable
    if torch.cuda.is_available():
        segmentor = segmentor.cuda()
        imgs = imgs.cuda()
        gt_semantic_seg = gt_semantic_seg.cuda()
    else:
        segmentor = _convert_batchnorm(segmentor)

    # Test forward train
    losses = segmentor.forward(
        imgs, img_metas, gt_semantic_seg=gt_semantic_seg, return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        segmentor.eval()
        # pack into lists
        img_list = [img[None, :] for img in imgs]
        img_meta_list = [[img_meta] for img_meta in img_metas]
        segmentor.forward(img_list, img_meta_list, return_loss=False)
