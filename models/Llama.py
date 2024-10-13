import torch
from torch import Tensor
from typing import Tuple, List
from .BaseModel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, SPIECE_UNDERLINE
from fastchat.model import get_conversation_template


__all__ = ["Llama2", "Llama3"]


class Llama2RemoveLastApparitionTokenizer(LlamaTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token = self.eos_token  # In llama2, pad token is actually EOS token

    def _tokenize(self, text, **kwargs):
        tokens = super()._tokenize(text, **kwargs)
        if len(tokens) > 0 and tokens[-1] == SPIECE_UNDERLINE:
            tokens.pop()
        return tokens


class Llama2(BaseModel):
    def __init__(self, llama_2_path="meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, *args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            llama_2_path, torch_dtype=dtype, trust_remote_code=True, use_auth_token=True
        ).eval()
        tokenizer = Llama2RemoveLastApparitionTokenizer.from_pretrained(
            llama_2_path, trust_remote_code=True, use_fast=False
        )
        conv = get_conversation_template("llama-2")

        super(Llama2, self).__init__(model, tokenizer, conv, *args, **kwargs)

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
        now_tokens = self.remove_apparition_token(now_tokens)
        usr_prompt_length = len(now_tokens)  # since apparition token

        self.conv.update_last_message(f"{usr_prompt} {adv_suffix}")
        # self.conv.get_prompt()这玩意会往最后加个空格，导致出现29871多一个
        # 但是concatenation又是我们手动自己做的，导致下一次做的时候没有这个空格，因此产生错位
        # 20241011时我换了另一个办法解决这个问题，直接remove掉所有的29871
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        now_tokens = self.remove_apparition_token(now_tokens)
        total_prompt_length = len(now_tokens)
        grad_slice = slice(usr_prompt_length, total_prompt_length)

        self.conv.append_message(self.conv.roles[1], "")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        now_tokens = self.remove_apparition_token(now_tokens)
        target_slice_left = len(now_tokens) + 1 - 1

        self.conv.update_last_message(f"{target}")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        now_tokens = self.remove_apparition_token(now_tokens)
        target_slice_right = len(now_tokens) - 1 - 1 - 1  # no last token
        target_slice = slice(target_slice_left, target_slice_right)  # for output logits
        loss_slice = slice(target_slice_left - 1, target_slice_right - 1)

        return torch.tensor(now_tokens, device=self.device), grad_slice, target_slice, loss_slice

    @staticmethod
    def remove_apparition_token(input_ids: List[int]) -> List[int]:
        """
        Since llama2 has a bug about an apparition token 29871
        https://github.com/huggingface/transformers/issues/26273
        we need to remove it.
        :param input_ids: L, a 1-D list
        :return: input ids, a list with 29871 removed
        """
        return [token for token in input_ids if token != 29871]


class Llama3(Llama2):
    def __init__(self, llama_3_path="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.float16, *args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(llama_3_path, torch_dtype=dtype, trust_remote_code=True).eval()
        # llama3 mistakely set padding token to one of the eot token, here we need to repair it to hugging face format
        model.generation_config.pad_token_id = 128001
        model.generation_config.eos_token_id = 128009
        tokenizer = AutoTokenizer.from_pretrained(llama_3_path, trust_remote_code=True, use_fast=False)
        conv = get_conversation_template("llama-3")
        super(Llama2, self).__init__(model, tokenizer, conv, *args, **kwargs)

    def get_prompt(self, usr_prompt, adv_suffix, target, *args, **kwargs) -> Tuple[Tensor, slice, slice, slice]:
        self.conv.messages.clear()
        self.conv.append_message(self.conv.roles[0], f"{usr_prompt}")
        # llama3 will add \n### at the end.
        # Since we still need to appendix adv_suffix, we shouldn't add \n### here.
        # Hence we use self.conv.get_prompt()[:-5] rather than self.conv.get_prompt()
        now_tokens = self.tokenizer(self.conv.get_prompt()[:-5], *args, **kwargs).input_ids
        usr_prompt_length = len(now_tokens)

        # llama3的词比较多，有时候会把空格也算进一个token中。
        # 这种情况下如果使用{usr_prompt} {adv_suffix}，那么每次decode的话adv_suffix里会带一个空格，前面又会带一个空格，空格就会越来越多
        # 我并没有改系统库的代码，因此相当于用{usr_prompt}{adv_suffix}攻击{usr_prompt} {adv_suffix}。
        # 这可能对迁移性造成问题。但至少比一边优化一边加空格要好
        # 如果不加空格，adv_suffix有可能吃掉前一个token。
        # 解决方案：禁止使用带空格的token+使用空格。这并不是为了增加迁移性。而是因为不这样连白盒都不行。
        self.conv.update_last_message(f"{usr_prompt} {adv_suffix} ")
        now_tokens = self.tokenizer(self.conv.get_prompt()[:-5], *args, **kwargs).input_ids
        total_prompt_length = len(now_tokens)
        grad_slice = slice(usr_prompt_length, total_prompt_length - 1)

        self.conv.append_message(self.conv.roles[1], "")
        now_tokens = self.tokenizer(self.conv.get_prompt()[:-5], *args, **kwargs).input_ids
        target_slice_left = len(now_tokens) + 1 - 1

        self.conv.update_last_message(f"{target}")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        target_slice_right = len(now_tokens) - 1 - 1 - 1  # no last token
        target_slice = slice(target_slice_left, target_slice_right)  # for output logits
        loss_slice = slice(target_slice_left - 1, target_slice_right - 1)

        return torch.tensor(now_tokens, device=self.device), grad_slice, target_slice, loss_slice
