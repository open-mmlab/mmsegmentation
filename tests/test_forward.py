"""
pytest tests/test_forward.py
"""
import copy
from os.path import dirname, exists, join

import numpy as np
import torch
from mmcv.utils.parrots_wrapper import SyncBatchNorm


def _get_config_directory():
    """ Find the predefined segmentor config directory """
    try:
        # Assume we are running in the source mmsegmentation repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmseg
        repo_dpath = dirname(dirname(mmseg.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """
    Load a configuration as a python module
    """
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_segmentor_cfg(fname):
    """
    Grab configs necessary to create a segmentor. These are deep copied to
    allow for safe modification of parameters without influencing other tests.

    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg))
    return model, train_cfg, test_cfg


def test_pspnet_forward():
    _test_encoder_decoder_forward('pspnet/psp_r50_8x2_200e_cityscapes.py')


def test_fcnnet_forward():
    _test_encoder_decoder_forward('fcnnet/fcn_r50_8x1_160e_cityscapes.py')


def test_deeplabv3_forward():
    _test_encoder_decoder_forward(
        'deeplabv3/deeplabv3_r101_8x1_os8_110e_cityscapes.py')


def test_deeplabv3plus_forward():
    _test_encoder_decoder_forward(
        'deeplabv3plus/deeplabv3plus_r101_os8_8x1_110e_cityscapes.py')


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
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


def _test_encoder_decoder_forward(cfg_file):
    model, train_cfg, test_cfg = _get_segmentor_cfg(cfg_file)
    model['pretrained'] = None
    test_cfg['mode'] = 'whole'

    from mmseg.models import build_segmentor
    segmentor = build_segmentor(model, train_cfg=train_cfg, test_cfg=test_cfg)
    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)

    num_classes = segmentor.decode_head.num_classes
    # batch_size=2 for BatchNorm
    input_shape = (2, 3, 713, 713)
    mm_inputs = _demo_mm_inputs(input_shape, num_classes=num_classes)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    gt_semantic_seg = mm_inputs['gt_semantic_seg']
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


def _demo_mm_inputs(input_shape=(1, 3, 256, 512),
                    num_classes=10):  # yapf: disable
    """
    Create a superset of inputs needed to run test or train batches.

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
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs
