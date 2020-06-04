import torch


def get_dataloader():
    if torch.__version__ == 'parrots':
        from torch.utils.data import DataLoader, PoolDataLoader
    else:
        from torch.utils.data import DataLoader
        PoolDataLoader = DataLoader
    return {"DataLoader": DataLoader, "PoolDataLoader": PoolDataLoader}
