import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from torch import Tensor


def collate_fn_by_padding(batch: List[Dict[str, Tensor]], global_max_length: int = 1024) -> Tuple[Tensor, Tensor]:
    # 1. truncate all instances to be smaller than global_max_length.
    batch = [{"input_ids": x["input_ids"][:global_max_length]} for x in batch]
    # 2. padding.
    max_length = max(len(x["input_ids"]) for x in batch)
    padded_batch = [
        torch.cat([x["input_ids"], torch.zeros(max_length - x["input_ids"].numel(), dtype=torch.int)], dim=0)
        for x in batch
    ]
    attention_mask = [
        torch.cat(
            [
                torch.ones(x["input_ids"].numel(), dtype=torch.int),
                torch.zeros(max_length - x["input_ids"].numel(), dtype=torch.int),
            ],
            dim=0,
        )
        for x in batch
    ]
    return torch.stack(padded_batch), torch.stack(attention_mask)


def ddp_loader(dataset: Dataset, batch_size=256, num_workers=8, pin_memory=True, collate_fn=None):
    sampler = torch.utils.data.DistributedSampler(
        dataset=dataset, rank=torch.distributed.get_rank(), num_replicas=torch.distributed.get_world_size()
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        shuffle=None,
        collate_fn=collate_fn,
    )
    return loader
