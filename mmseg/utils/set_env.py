# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmseg into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmseg default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmseg`, and all registries will build modules from mmseg's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmseg.datasets  # noqa: F401,F403
    import mmseg.engine  # noqa: F401,F403
    import mmseg.evaluation  # noqa: F401,F403
    import mmseg.models  # noqa: F401,F403
    import mmseg.structures  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmseg')
        if never_created:
            DefaultScope.get_instance('mmseg', scope_name='mmseg')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmseg':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmseg", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmseg". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmseg-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmseg')
