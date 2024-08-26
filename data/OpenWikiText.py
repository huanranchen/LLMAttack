import os
import torch
from typing import List, Dict
from torch import Tensor, Tuple
from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

__all__ = ["get_open_wiki_text_loader", "get_open_wiki_text_loader_for_train"]


def get_open_wiki_text_loader(batch_size=1, num_workers=8, shuffle=True):
    dataset = load_dataset("openwebtext", cache_dir="./resources/openwebtext/")["train"]
    dataset = dataset.with_format("torch")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader


def collate_fn_by_padding(batch: List[Dict[str, Tensor]], global_max_length: int = 1024) -> Tuple[Tensor, Tensor]:
    # 1. truncate all instances to be smaller than global_max_length.
    batch = [{"input_ids": x["input_ids"][:global_max_length]} for x in batch]
    # 2. padding.
    max_length = max(len(x["input_ids"]) for x in batch)
    padded_batch = [
        torch.cat(
            [x["input_ids"], torch.zeros(max_length - x["input_ids"].numel(), dtype=torch.int)],
            dim=0,
        )
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


def get_open_wiki_text_loader_for_train(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    num_workers: int = 8,
    shuffle: bool = True,
    saving_path: str = "./resources/openwebtext/gpt2_tokenized_version/",
) -> DataLoader:
    def preprocess_and_tokenize(example):
        text = example["text"]
        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token
        tokens["input_ids"].append(tokenizer.eos_token_id)
        return tokens

    if os.path.exists(saving_path):
        dataset = load_from_disk(saving_path)
    else:
        dataset = load_dataset("openwebtext", cache_dir="./resources/openwebtext/")["train"]
        dataset = dataset.map(preprocess_and_tokenize)
        # dataset.save_to_disk(saving_path)
    dataset = dataset.with_format("torch")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn_by_padding,
    )
    return loader
