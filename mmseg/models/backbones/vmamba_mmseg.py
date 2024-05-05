import os
from functools import partial
from typing import Callable

import torch
from torch import nn
from torch.utils import checkpoint

from mmengine.model import BaseModule
# from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

build = import_abspy(
    "models", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)





Backbone_VSSM: nn.Module = build.vmamba.Backbone_VSSM

@MODELS_MMSEG.register_module()
# @MODELS_MMDET.register_module()
class MM_VSSM(BaseModule, Backbone_VSSM):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_VSSM.__init__(self, *args, **kwargs)


# ===============================================
from typing import Union, Tuple, Any
selective_scan_flop_jit: Callable = build.vmamba.selective_scan_flop_jit

supported_extra_ops={
    "aten::silu": None, # as relu is in _IGNORED_OPS
    "aten::neg": None, # as relu is in _IGNORED_OPS
    "aten::exp": None, # as relu is in _IGNORED_OPS
    "aten::flip": None, # as permute is in _IGNORED_OPS
    "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScan": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit, # latter
}


def mmengine_flop_count(model: nn.Module = None, input_shape = (3, 224, 224), show_table=False, show_arch=False, _get_model_complexity_info=False):
    from mmengine.analysis.print_helper import is_tuple_of, FlopAnalyzer, ActivationAnalyzer, parameter_count, _format_size, complexity_stats_table, complexity_stats_str
    from mmengine.analysis.jit_analysis import _IGNORED_OPS
    from mmengine.analysis.complexity_analysis import _DEFAULT_SUPPORTED_FLOP_OPS, _DEFAULT_SUPPORTED_ACT_OPS
    from mmengine.analysis import get_model_complexity_info as mm_get_model_complexity_info
    
    # modified from mmengine.analysis
    def get_model_complexity_info(
        model: nn.Module,
        input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...],
                        None] = None,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[Any, ...],
                    None] = None,
        show_table: bool = True,
        show_arch: bool = True,
    ):
        if input_shape is None and inputs is None:
            raise ValueError('One of "input_shape" and "inputs" should be set.')
        elif input_shape is not None and inputs is not None:
            raise ValueError('"input_shape" and "inputs" cannot be both set.')

        if inputs is None:
            device = next(model.parameters()).device
            if is_tuple_of(input_shape, int):  # tuple of int, construct one tensor
                inputs = (torch.randn(1, *input_shape).to(device), )
            elif is_tuple_of(input_shape, tuple) and all([
                    is_tuple_of(one_input_shape, int)
                    for one_input_shape in input_shape  # type: ignore
            ]):  # tuple of tuple of int, construct multiple tensors
                inputs = tuple([
                    torch.randn(1, *one_input_shape).to(device)
                    for one_input_shape in input_shape  # type: ignore
                ])
            else:
                raise ValueError(
                    '"input_shape" should be either a `tuple of int` (to construct'
                    'one input tensor) or a `tuple of tuple of int` (to construct'
                    'multiple input tensors).')

        flop_handler = FlopAnalyzer(model, inputs).set_op_handle(**supported_extra_ops)
        # activation_handler = ActivationAnalyzer(model, inputs)

        flops = flop_handler.total()
        # activations = activation_handler.total()
        params = parameter_count(model)['']

        flops_str = _format_size(flops)
        # activations_str = _format_size(activations)
        params_str = _format_size(params)

        if show_table:
            complexity_table = complexity_stats_table(
                flops=flop_handler,
                # activations=activation_handler,
                show_param_shapes=True,
            )
            complexity_table = '\n' + complexity_table
        else:
            complexity_table = ''

        if show_arch:
            complexity_arch = complexity_stats_str(
                flops=flop_handler,
                # activations=activation_handler,
            )
            complexity_arch = '\n' + complexity_arch
        else:
            complexity_arch = ''

        return {
            'flops': flops,
            'flops_str': flops,
            # 'activations': activations,
            # 'activations_str': activations_str,
            'params': params,
            'params_str': params,
            'out_table': complexity_table,
            'out_arch': complexity_arch
        }
    
    if _get_model_complexity_info:
        return get_model_complexity_info

    model.eval()
    analysis_results = get_model_complexity_info(
        model,
        input_shape,
        show_table=show_table,
        show_arch=show_arch,
    )
    flops = analysis_results['flops_str']
    params = analysis_results['params_str']
    # activations = analysis_results['activations_str']
    out_table = analysis_results['out_table']
    out_arch = analysis_results['out_arch']
    
    if show_arch:
        print(out_arch)
    
    if show_table:
        print(out_table)
    
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\t'
          f'Flops: {flops}\tParams: {params}\t'
        #   f'Activation: {activations}\n{split_line}'
    , flush=True)
    # print('!!!Only the backbone network is counted in FLOPs analysis.')
    # print('!!!Please be cautious if you use the results in papers. '
    #       'You may need to check if all ops are supported and verify that the '
    #       'flops computation is correct.')


def mmseg_flops(config=None, input_shape=(3, 512, 2048)):
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(config)
    cfg["work_dir"] = "/tmp"
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    
    info = mmengine_flop_count(model, input_shape=input_shape)


def mmdet_flops(config=None):
    from mmengine.config import Config
    from mmengine.runner import Runner
    import numpy as np
    import os

    cfg = Config.fromfile(config)
    cfg["work_dir"] = "/tmp"
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    get_model_complexity_info = mmengine_flop_count(_get_model_complexity_info=True)
    
    if True:
        oridir = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), "../detection"))
        data_loader = runner.val_dataloader
        num_images = 100
        mean_flops = []
        for idx, data_batch in enumerate(data_loader):
            if idx == num_images:
                break
            data = model.data_preprocessor(data_batch)
            model.forward = partial(model.forward, data_samples=data['data_samples'])
            # out = get_model_complexity_info(model, inputs=data['inputs'])
            out = get_model_complexity_info(model, input_shape=(3, 1280, 800))
            params = out['params_str']
            mean_flops.append(out['flops'])
        mean_flops = np.average(np.array(mean_flops))
        print(params, mean_flops)
        os.chdir(oridir)


if __name__ == "__main__":
    this_path = os.path.dirname(os.path.abspath(__file__)).rstrip("/")
    mmdet_path = this_path + "/../detection"
    mmseg_path = this_path + "/../segmentation"
    # mmdet_flops(f"{mmdet_path}/configs/vssm/mask_rcnn_vssm_fpn_coco_tiny.py") # 42.4M 285883020640.0
    # mmdet_flops(f"{mmdet_path}/configs/vssm/mask_rcnn_vssm_fpn_coco_small.py") # 63.924M 400260276640.0
    # mmdet_flops(f"{mmdet_path}/configs/vssm/mask_rcnn_vssm_fpn_coco_base.py") # 95.628M 539797328640.0

    # mmseg_flops(f"{mmseg_path}/configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py") # Flops: 963853165152.0   Params: 54546956
    # mmseg_flops(f"{mmseg_path}/configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py") #  Flops: 1080975662496.0  Params: 76070924
    # mmseg_flops(f"{mmseg_path}/configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py") # Flops: 1225941913504.0  Params: 109765548
    # mmseg_flops(f"{mmseg_path}/configs/vssm/upernet_vssm_4xb4-160k_ade20k-640x640_small.py", input_shape=(3, 640, 2560)) # Flops: 1689017528400.0  Params: 76070924





