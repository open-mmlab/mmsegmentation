# Modified from https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend  # noqa

import torch
import torch.nn.functional as F


def normalize(grid):
    """Normalize input grid from [-1, 1] to [0, 1]
    Args:
        grid (Tensor): The grid to be normalize, range [-1, 1].
    Returns:
        Tensor: Normalized grid, range [0, 1].
    """

    return (grid + 1.0) / 2.0


def denormalize(grid):
    """Denormalize input grid from range [0, 1] to [-1, 1]
    Args:
        grid (Tensor): The grid to be denormalize, range [0, 1].
    Returns:
        Tensor: Denormalized grid, range [-1, 1].
    """

    return grid * 2.0 - 1.0


def generate_grid(num_grid, size, device):
    """Generate regular square grid of points in [0, 1] x [0, 1] coordinate
    space.

    Args:
        num_grid (int): The number of grids to sample, one for each region.
        size (tuple(int, int)): The side size of the regular grid.
        device (torch.device): Desired device of returned tensor.

    Returns:
        (torch.Tensor): A tensor of shape (num_grid, size[0]*size[1], 2) that
            contains coordinates for the regular grids.
    """

    affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], device=device)
    grid = F.affine_grid(
        affine_trans, torch.Size((1, 1, *size)), align_corners=False)
    grid = normalize(grid)
    return grid.view(1, -1, 2).expand(num_grid, -1, -1)


def point_sample(input, points, align_corners=False, **kwargs):
    """A wrapper around :func:`grid_sample` to support 3D point_coords tensors
    Unlike :func:`torch.nn.functional.grid_sample` it assumes point_coords to
    lie inside ``[0, 1] x [0, 1]`` square.

    Args:
        input (Tensor): Feature map, shape (N, C, H, W).
        points (Tensor): Image based absolute point coordinates (normalized),
            range [0, 1] x [0, 1], shape (N, P, 2) or (N, Hgrid, Wgrid, 2).
        align_corners (bool): Whether align_corners. Default: False

    Returns:
        Tensor: Features of `point` on `input`, shape (N, C, P) or
            (N, C, Hgrid, Wgrid).
    """

    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output
