import torch
from torch import Tensor
from typing import Tuple
from .BaseModel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from fastchat.model import get_conversation_template


__all__ = ["Llama2", "Llama3"]


class Llama2(BaseModel):
    def __init__(
        self,
        llama_2_path=(
            "/workspace/home/chenhuanran2022/.cache/huggingface/hub/"
            "models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/"
        ),
    ):
        model = AutoModelForCausalLM.from_pretrained(
            llama_2_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
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
        now_tokens = self.tokenizer(self.conv.get_prompt()).input_ids
        total_prompt_length = len(now_tokens)  # since apparition token
        grad_slice = slice(usr_prompt_length, total_prompt_length)

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
        llama_3_path="/workspace/home/chenhuanran2022/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6/",
    ):
        model = AutoModelForCausalLM.from_pretrained(
            llama_3_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(llama_3_path, trust_remote_code=True, use_fast=False)
        conv = get_conversation_template("llama-3")
        super(Llama2, self).__init__(model, tokenizer, conv)

    def get_prompt(self, usr_prompt, adv_suffix, target) -> Tuple[Tensor, slice, slice, slice]:
        self.conv.messages.clear()
        self.conv.append_message(self.conv.roles[0], f"{usr_prompt}")
        now_tokens = self.tokenizer(self.conv.get_prompt()).input_ids
        usr_prompt_length = len(now_tokens)

        self.conv.update_last_message(f"{usr_prompt} {adv_suffix}")
        now_tokens = self.tokenizer(self.conv.get_prompt()).input_ids
        total_prompt_length = len(now_tokens)
        # To remove \n ### and space.
        grad_slice = slice(usr_prompt_length - 2, total_prompt_length - 3)

        self.conv.append_message(self.conv.roles[1], "")
        now_tokens = self.tokenizer(self.conv.get_prompt()).input_ids
        target_slice_left = len(now_tokens) + 1 - 1

        self.conv.update_last_message(f"{target}")
        now_tokens = self.tokenizer(self.conv.get_prompt()).input_ids
        target_slice_right = len(now_tokens) - 1 - 1 - 1  # no last token
        target_slice = slice(target_slice_left, target_slice_right)  # for output logits
        loss_slice = slice(target_slice_left - 1, target_slice_right - 1)
        return torch.tensor(now_tokens, device=self.device), grad_slice, target_slice, loss_slice
