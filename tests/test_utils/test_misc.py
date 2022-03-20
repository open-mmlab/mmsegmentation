# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

from mmseg.utils import find_latest_checkpoint


def test_find_latest_checkpoint():
    with tempfile.TemporaryDirectory() as tempdir:
        # no checkpoints in the path
        path = tempdir
        latest = find_latest_checkpoint(path)
        assert latest is None

        # The path doesn't exist
        path = osp.join(tempdir, 'none')
        latest = find_latest_checkpoint(path)
        assert latest is None

    # test when latest.pth exists
    with tempfile.TemporaryDirectory() as tempdir:
        with open(osp.join(tempdir, 'latest.pth'), 'w') as f:
            f.write('latest')
        path = tempdir
        latest = find_latest_checkpoint(path)
        assert latest == osp.join(tempdir, 'latest.pth')

    with tempfile.TemporaryDirectory() as tempdir:
        for iter in range(1600, 160001, 1600):
            with open(osp.join(tempdir, f'iter_{iter}.pth'), 'w') as f:
                f.write(f'iter_{iter}.pth')
        latest = find_latest_checkpoint(tempdir)
        assert latest == osp.join(tempdir, 'iter_160000.pth')

    with tempfile.TemporaryDirectory() as tempdir:
        for epoch in range(1, 21):
            with open(osp.join(tempdir, f'epoch_{epoch}.pth'), 'w') as f:
                f.write(f'epoch_{epoch}.pth')
        latest = find_latest_checkpoint(tempdir)
        assert latest == osp.join(tempdir, 'epoch_20.pth')
