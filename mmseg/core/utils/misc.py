# Copyright (c) OpenMMLab. All rights reserved.
def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs
