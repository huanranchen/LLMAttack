import torch
from torch import Tensor
from typing import Tuple
from .BaseModel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from fastchat.model import get_conversation_template


__all__ = ["Llama2", "Llama3"]


class Llama2(BaseModel):
    def __init__(
        self,
        llama_2_path="meta-llama/Llama-2-7b-chat-hf",
        dtype=torch.float16,
    ):
        model = AutoModelForCausalLM.from_pretrained(
            llama_2_path, torch_dtype=dtype, trust_remote_code=True, use_auth_token=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(llama_2_path, trust_remote_code=True, use_fast=False)
        conv = get_conversation_template("llama-2")
        super(Llama2, self).__init__(model, tokenizer, conv)

    def get_prompt(self, usr_prompt, adv_suffix, target) -> Tuple[Tensor, slice, slice, slice]:
        """
        Llama 2 has a bug (29871 apparition token)
        :param usr_prompt:
        :param adv_suffix:
        :param target:
        :return: input_ids, grad_slice, target_slice, loss_slice (which corresponding output need calculate loss)
        """
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


class Llama3(Llama2):
    def __init__(
        self,
        llama_3_path="meta-llama/Meta-Llama-3-8B-Instruct",
        dtype=torch.float16,
    ):
        model = AutoModelForCausalLM.from_pretrained(
            llama_3_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).eval()
        # llama3 mistakely set padding token to one of the eot token, here we need to repair it to hugging face format
        model.generation_config.pad_token_id = 128001
        model.generation_config.eos_token_id = 128009
        tokenizer = AutoTokenizer.from_pretrained(llama_3_path, trust_remote_code=True, use_fast=False)
        conv = get_conversation_template("llama-3")
        super(Llama2, self).__init__(model, tokenizer, conv)

    def get_prompt(self, usr_prompt, adv_suffix, target) -> Tuple[Tensor, slice, slice, slice]:
        self.conv.messages.clear()
        self.conv.append_message(self.conv.roles[0], f"{usr_prompt}")
        # llama3 will add \n### at the end.
        # Since we still need to appendix adv_suffix, we shouldn't add \n### here.
        # Hence we use self.conv.get_prompt()[:-5] rather than self.conv.get_prompt()
        now_tokens = self.tokenizer(self.conv.get_prompt()[:-5]).input_ids
        usr_prompt_length = len(now_tokens)

        self.conv.update_last_message(f"{usr_prompt} {adv_suffix}")
        now_tokens = self.tokenizer(self.conv.get_prompt()[:-5]).input_ids
        total_prompt_length = len(now_tokens)
        grad_slice = slice(usr_prompt_length + 1, total_prompt_length)

        self.conv.append_message(self.conv.roles[1], "")
        now_tokens = self.tokenizer(self.conv.get_prompt()[:-5]).input_ids
        target_slice_left = len(now_tokens) + 1 - 1

        self.conv.update_last_message(f"{target}")
        now_tokens = self.tokenizer(self.conv.get_prompt()).input_ids
        target_slice_right = len(now_tokens) - 1 - 1 - 1  # no last token
        target_slice = slice(target_slice_left, target_slice_right)  # for output logits
        loss_slice = slice(target_slice_left - 1, target_slice_right - 1)

        return torch.tensor(now_tokens, device=self.device), grad_slice, target_slice, loss_slice
