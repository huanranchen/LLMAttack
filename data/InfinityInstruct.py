import os
import torch
import random
from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from .utils import ddp_loader, collate_fn_by_padding

__all__ = ["get_infinity_instruct_loader", "get_infinity_instruct_loader_for_train"]


def get_infinity_instruct_loader(batch_size=1, num_workers=8, shuffle=True):
    dataset = load_dataset("BAAI/Infinity-Instruct", "Gen", split="train")["conversation"]
    dataset = dataset.with_format("torch")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader


def get_infinity_instruct_loader_for_train(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    num_workers: int = 8,
    shuffle: bool = True,
    saving_path: str = "./resources/InfinityInstruct/gpt2_tokenized_version/",
    ddp: bool = False,
) -> DataLoader:
    def preprocess_and_tokenize(example):
        text = example["conversations"]
        text_a = "### Human: " + text[0]["value"] + "\n### Assistant: " + text[1]["value"]
        text_b = text[0]["value"]
        text_c = text[1]["value"]
        text_d = "### Human: " + text[0]["value"]
        text_e = "\n### Assistant: " + text[1]["value"]
        # text = random.choice([text_a, text_b, text_c, text_d, text_e])
        text = text_a
        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token
        tokens["input_ids"].append(tokenizer.eos_token_id)
        return tokens

    if os.path.exists(saving_path):
        dataset = load_from_disk(saving_path)
    else:
        dataset = load_dataset("BAAI/Infinity-Instruct", "Gen", split="train")
        dataset = dataset.map(preprocess_and_tokenize)
        dataset.save_to_disk(saving_path)
    dataset = dataset.with_format("torch")

    if not ddp:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn_by_padding,
        )
        return loader
    return ddp_loader(dataset, batch_size, num_workers, collate_fn=collate_fn_by_padding)
