from mmseg.models.utils import make_divisible


def test_make_divisible():
    # test with min_value = None
    assert make_divisible(10, 4) == 12
    assert make_divisible(9, 4) == 12
    assert make_divisible(1, 4) == 4

    # test with min_value = 8
    assert make_divisible(10, 4, 8) == 12
    assert make_divisible(9, 4, 8) == 12
    assert make_divisible(1, 4, 8) == 8
