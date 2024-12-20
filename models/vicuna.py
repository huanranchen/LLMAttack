import torch
from torch import Tensor
from typing import Tuple
from .BaseModel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from fastchat.model import get_conversation_template


__all__ = ["Vicuna15"]


class Vicuna15(BaseModel):
    def __init__(
        self,
        vicuna_15_path="lmsys/vicuna-7b-v1.5",
        conv=get_conversation_template("stable-vicuna"),
        dtype=torch.float16,
        *args,
        **kwargs,
    ):
        model = AutoModelForCausalLM.from_pretrained(
            vicuna_15_path, torch_dtype=dtype, trust_remote_code=True, use_auth_token=True, top_p=None
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(vicuna_15_path, trust_remote_code=True, use_fast=False)
        super(Vicuna15, self).__init__(model, tokenizer, conv, *args, **kwargs)

    def generate(self, *args, **kwargs) -> str or Tuple[str, Tensor]:
        result = super(Vicuna15, self).generate(*args, **kwargs)
        if type(result) is tuple:
            return result
        # 对result是str后处理。防止模型一直### Human, ###Assitant, ### Human这样自问自答
        return result.split("###")[0]

    def get_prompt(self, usr_prompt, adv_suffix, target, *args, **kwargs) -> Tuple[Tensor, slice, slice, slice]:
        """
        :param usr_prompt:
        :param adv_suffix:
        :param target:
        :return: input_ids, grad_slice, target_slice, loss_slice (which corresponding output need calculate loss)
        """
        self.conv.messages.clear()
        self.conv.append_message(self.conv.roles[0], f"{usr_prompt}")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        usr_prompt_length = len(now_tokens)  # since apparition token

        self.conv.update_last_message(f"{usr_prompt} {adv_suffix}")

        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        total_prompt_length = len(now_tokens)
        grad_slice = slice(usr_prompt_length - 1, total_prompt_length - 1)

        self.conv.append_message(self.conv.roles[1], "")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        target_slice_left = len(now_tokens) + 1 - 1

        self.conv.update_last_message(f"{target}")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        target_slice_right = len(now_tokens) - 1 - 1 - 1  # no last token
        target_slice = slice(target_slice_left, target_slice_right)  # for output logits
        loss_slice = slice(target_slice_left - 1, target_slice_right - 1)

        return torch.tensor(now_tokens, device=self.device), grad_slice, target_slice, loss_slice
