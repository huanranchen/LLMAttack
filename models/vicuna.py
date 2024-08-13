import torch
from torch import Tensor
from typing import Tuple
from .BaseModel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from fastchat.model import get_conversation_template


__all__ = ["Vicuna15"]


class Vicuna15(BaseModel):
    def __init__(self, vicuna_15_path="lmsys/vicuna-7b-v1.5", dtype=torch.float32):
        model = AutoModelForCausalLM.from_pretrained(
            vicuna_15_path, torch_dtype=dtype, trust_remote_code=True, use_auth_token=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(vicuna_15_path, trust_remote_code=True, use_fast=False)
        conv = get_conversation_template("stable-vicuna")
        super(Vicuna15, self).__init__(model, tokenizer, conv)

    def get_prompt(self, usr_prompt, adv_suffix, target) -> Tuple[Tensor, slice, slice, slice]:
        self.conv.messages.clear()
        self.conv.append_message(self.conv.roles[0], f"{usr_prompt}")
        now_tokens = self.tokenizer(self.conv.get_prompt()).input_ids
        usr_prompt_length = len(now_tokens)  # since apparition token

        self.conv.update_last_message(f"{usr_prompt} {adv_suffix}")
        # self.conv.get_prompt()这玩意会往最后加个空格，导致出现29871多一个
        # 但是concatenation又是我们手动自己做的，导致下一次做的时候没有这个空格，因此产生错位
        now_tokens = self.tokenizer(self.conv.get_prompt()).input_ids
        total_prompt_length = len(now_tokens)
        grad_slice = slice(usr_prompt_length - 1, total_prompt_length - 1)

        self.conv.append_message(self.conv.roles[1], "")
        now_tokens = self.tokenizer(self.conv.get_prompt()).input_ids
        target_slice_left = len(now_tokens) + 1 - 1

        self.conv.update_last_message(f"{target}")
        now_tokens = self.tokenizer(self.conv.get_prompt()).input_ids
        target_slice_right = len(now_tokens) - 1 - 1 - 1  # no last token
        target_slice = slice(target_slice_left, target_slice_right)  # for output logits
        loss_slice = slice(target_slice_left - 1, target_slice_right - 1)

        return torch.tensor(now_tokens, device=self.device), grad_slice, target_slice, loss_slice
