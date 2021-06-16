from mmseg.models.utils import (to_1tuple, to_2tuple, to_3tuple, to_4tuple,
                                to_ntuple)


def test_to_xtuple():
    # to_xtuple will convert int to tuple.
    num = 224

    assert to_1tuple(num) == (224, )
    assert to_2tuple(num) == (224, 224)
    assert to_3tuple(num) == (224, 224, 224)
    assert to_4tuple(num) == (224, 224, 224, 224)
    assert to_ntuple(5)(num) == (224, 224, 224, 224, 224)
