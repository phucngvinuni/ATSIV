# code/classification/collate.py
import torch

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)