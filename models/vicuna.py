import torch
from .Llama import Llama2
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from fastchat.model import get_conversation_template


__all__ = ["Vicuna15"]


class Vicuna15(Llama2):
    def __init__(self, vicuna_15_path="lmsys/vicuna-7b-v1.5", dtype=torch.float16, *args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            vicuna_15_path, torch_dtype=dtype, trust_remote_code=True, use_auth_token=True, top_p=None
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(vicuna_15_path, trust_remote_code=True, use_fast=False)
        conv = get_conversation_template("stable-vicuna")
        super(Llama2, self).__init__(model, tokenizer, conv, *args, **kwargs)
