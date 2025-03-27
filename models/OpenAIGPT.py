import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple
from fastchat.conversation import get_conv_template
from .BaseModel import BaseModel

openai_all_gpts = [
    "openai-community/openai-gpt",
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
]

__all__ = ["OpenAIGPT", "OpenAIGPTWithChatTemplate"]


class OpenAIGPT(BaseModel):
    def __init__(self, model_name: str = "openai-community/openai-gpt", *args, **kwargs):
        assert model_name in openai_all_gpts
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        conv = get_conv_template("raw")
        super(OpenAIGPT, self).__init__(model, tokenizer, conv, *args, **kwargs)

    def get_prompt(self, *args, **kwargs):
        return self.get_prompt_raw(*args, **kwargs)


class OpenAIGPTWithChatTemplate(BaseModel):
    def __init__(self, model_name: str = "openai-community/openai-gpt", dtype=torch.float16, *args, **kwargs):
        # assert model_name in openai_all_gpts
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        conv = get_conv_template("vicuna_v1.1")  # personally I like this template most.
        # I also believe an only pretrained model can be able to do conversation using this template
        super(OpenAIGPTWithChatTemplate, self).__init__(model, tokenizer, conv, *args, **kwargs)

    def get_prompt(self, usr_prompt, adv_suffix, target, *args, **kwargs) -> Tuple[Tensor, slice, slice, slice]:
        self.conv.messages.clear()
        self.conv.append_message(self.conv.roles[0], f"{usr_prompt}")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        usr_prompt_length = len(now_tokens) - 1  # "vicuna_v1.1"模板会在末尾加空格

        self.conv.update_last_message(f"{usr_prompt} {adv_suffix}")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        total_prompt_length = len(now_tokens) - 1  # "vicuna_v1.1"模板会在末尾加空格
        grad_slice = slice(usr_prompt_length, total_prompt_length)

        self.conv.append_message(self.conv.roles[1], "")  # roles结尾不加空格
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        target_slice_left = len(now_tokens)
        self.conv.update_last_message(f"{target}")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids

        target_slice_right = len(now_tokens) - 1 - 1 - 1  # 去掉<eos>
        target_slice = slice(target_slice_left, target_slice_right)
        loss_slice = slice(target_slice_left - 1, target_slice_right - 1)

        return torch.tensor(now_tokens, device=self.device), grad_slice, target_slice, loss_slice
