import os
from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from .utils import ddp_loader, collate_fn_by_padding

__all__ = ["get_open_wiki_text_loader", "get_open_wiki_text_loader_for_train"]


def get_open_wiki_text_loader(batch_size=1, num_workers=8, shuffle=True):
    dataset = load_dataset("openwebtext", cache_dir="./resources/openwebtext/")["train"]
    dataset = dataset.with_format("torch")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader


def get_open_wiki_text_loader_for_train(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    num_workers: int = 8,
    shuffle: bool = True,
    saving_path: str = "./resources/openwebtext/gpt2_tokenized_version/",
    ddp: bool = False,
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
