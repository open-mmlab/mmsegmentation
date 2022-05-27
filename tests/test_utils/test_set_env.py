# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import multiprocessing as mp
import os
import platform
import sys
from unittest import TestCase

import cv2
import pytest
from mmcv import Config
from mmengine import DefaultScope

from mmseg.utils import register_all_modules, setup_multi_processes


@pytest.mark.parametrize('workers_per_gpu', (0, 2))
@pytest.mark.parametrize(('valid', 'env_cfg'), [(True,
                                                 dict(
                                                     mp_start_method='fork',
                                                     opencv_num_threads=0,
                                                     omp_num_threads=1,
                                                     mkl_num_threads=1)),
                                                (False,
                                                 dict(
                                                     mp_start_method=1,
                                                     opencv_num_threads=0.1,
                                                     omp_num_threads='s',
                                                     mkl_num_threads='1'))])
def test_setup_multi_processes(workers_per_gpu, valid, env_cfg):
    # temp save system setting
    sys_start_mehod = mp.get_start_method(allow_none=True)
    sys_cv_threads = cv2.getNumThreads()
    # pop and temp save system env vars
    sys_omp_threads = os.environ.pop('OMP_NUM_THREADS', default=None)
    sys_mkl_threads = os.environ.pop('MKL_NUM_THREADS', default=None)

    config = dict(data=dict(workers_per_gpu=workers_per_gpu))
    config.update(env_cfg)
    cfg = Config(config)
    setup_multi_processes(cfg)

    # test when cfg is valid and workers_per_gpu > 0
    # setup_multi_processes will work
    if valid and workers_per_gpu > 0:
        # test config without setting env

        assert os.getenv('OMP_NUM_THREADS') == str(env_cfg['omp_num_threads'])
        assert os.getenv('MKL_NUM_THREADS') == str(env_cfg['mkl_num_threads'])
        # when set to 0, the num threads will be 1
        assert cv2.getNumThreads() == env_cfg[
            'opencv_num_threads'] if env_cfg['opencv_num_threads'] > 0 else 1
        if platform.system() != 'Windows':
            assert mp.get_start_method() == env_cfg['mp_start_method']

        # revert setting to avoid affecting other programs
        if sys_start_mehod:
            mp.set_start_method(sys_start_mehod, force=True)
        cv2.setNumThreads(sys_cv_threads)
        if sys_omp_threads:
            os.environ['OMP_NUM_THREADS'] = sys_omp_threads
        else:
            os.environ.pop('OMP_NUM_THREADS')
        if sys_mkl_threads:
            os.environ['MKL_NUM_THREADS'] = sys_mkl_threads
        else:
            os.environ.pop('MKL_NUM_THREADS')

    elif valid and workers_per_gpu == 0:

        if platform.system() != 'Windows':
            assert mp.get_start_method() == env_cfg['mp_start_method']
        assert cv2.getNumThreads() == env_cfg[
            'opencv_num_threads'] if env_cfg['opencv_num_threads'] > 0 else 1
        assert 'OMP_NUM_THREADS' not in os.environ
        assert 'MKL_NUM_THREADS' not in os.environ
        if sys_start_mehod:
            mp.set_start_method(sys_start_mehod, force=True)
        cv2.setNumThreads(sys_cv_threads)
        if sys_omp_threads:
            os.environ['OMP_NUM_THREADS'] = sys_omp_threads
        if sys_mkl_threads:
            os.environ['MKL_NUM_THREADS'] = sys_mkl_threads

    else:
        assert mp.get_start_method() == sys_start_mehod
        assert cv2.getNumThreads() == sys_cv_threads
        assert 'OMP_NUM_THREADS' not in os.environ
        assert 'MKL_NUM_THREADS' not in os.environ


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from mmseg.registry import DATASETS

        # not init default scope
        sys.modules.pop('mmseg.datasets', None)
        sys.modules.pop('mmseg.datasets.ade', None)
        DATASETS._module_dict.pop('ADE20KDataset', None)
        self.assertFalse('ADE20KDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue('ADE20KDataset' in DATASETS.module_dict)

        # init default scope
        sys.modules.pop('mmseg.datasets')
        sys.modules.pop('mmseg.datasets.ade')
        DATASETS._module_dict.pop('ADE20KDataset', None)
        self.assertFalse('ADE20KDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue('ADE20KDataset' in DATASETS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'mmseg')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning, 'The current default scope "test" is not "mmseg"'):
            register_all_modules(init_default_scope=True)
