# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import sys
from unittest import TestCase

from mmengine import DefaultScope

from mmseg.utils import register_all_modules


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
