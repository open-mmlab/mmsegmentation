import torch
import torch.nn as nn


class Readout(nn.Module):

    def __init__(self, start_index=1):
        super(Readout, self).__init__()
        self.start_index = start_index


class Slice(Readout):

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(Readout):

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class ProjectReadout(Readout):

    def __init__(self, in_channels, start_index=1):
        super().__init__(start_index=start_index)
        self.project = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels), nn.GELU)

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)
        return self.project(features)


def _make_readout_ops(channels, out_channels, readout_type, start_index):
    if readout_type == 'ignore':
        readout_ops = [Slice(start_index) for _ in out_channels]
    elif readout_type == 'add':
        readout_ops = [AddReadout(start_index) for _ in out_channels]
    elif readout_type == 'project':
        readout_ops = [
            ProjectReadout(channels, start_index) for _ in out_channels
        ]
    else:
        assert f"unexpected readout operation type, expected 'ignore',\
            'add' or 'project', but got {readout_type}"

    return readout_ops
