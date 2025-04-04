import torch
from torch import nn, Tensor
from transformers import PreTrainedTokenizer, PreTrainedModel
from fastchat.conversation import Conversation
from typing import Tuple


class BaseModel(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizer = None,
        conv: Conversation = None,
        generation_max_length: int = 300,
        device=torch.device("cuda"),
    ):
        super(BaseModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.conv = conv
        self.eval().requires_grad_(False).to(device)
        self.device = device
        self.generation_max_length = generation_max_length
        self.config = model.config
        self.tie_weights = self.model.tie_weights

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_prompt(self, usr_prompt, adv_suffix, target, *args, **kwargs) -> Tuple[Tensor, slice, slice, slice]:
        """
        Llama 2 has a bug (29871 apparition token)
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
        total_prompt_length = len(now_tokens)  # since apparition token
        grad_slice = slice(usr_prompt_length, total_prompt_length)

        self.conv.append_message(self.conv.roles[1], "")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        target_slice_left = len(now_tokens) + 1 - 1

        self.conv.update_last_message(f"{target}")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        target_slice_right = len(now_tokens) - 1 - 1 - 1  # no last token
        target_slice = slice(target_slice_left, target_slice_right)  # for output logits
        loss_slice = slice(target_slice_left - 1, target_slice_right - 1)
        return torch.tensor(now_tokens, device=self.device), grad_slice, target_slice, loss_slice

    def get_prompt_raw(self, usr_prompt, adv_suffix, target, *args, **kwargs) -> Tuple[Tensor, slice, slice, slice]:
        """
        :param usr_prompt:
        :param adv_suffix:
        :param target:
        :return: input_ids, grad_slice, target_slice, loss_slice (which corresponding output need calculate loss)
        """
        self.conv.messages.clear()
        self.conv.append_message(self.conv.roles[0], f"{usr_prompt}")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        usr_prompt_length = len(now_tokens)

        self.conv.update_last_message(f"{usr_prompt} {adv_suffix}")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        total_prompt_length = len(now_tokens)
        grad_slice = slice(usr_prompt_length, total_prompt_length)

        self.conv.append_message(self.conv.roles[1], "")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        target_slice_left = len(now_tokens)

        self.conv.update_last_message(f"{target}")
        now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
        target_slice_right = len(now_tokens)
        target_slice = slice(target_slice_left, target_slice_right)
        loss_slice = slice(target_slice_left - 1, target_slice_right - 1)

        return torch.tensor(now_tokens, device=self.device), grad_slice, target_slice, loss_slice

    def generate(self, question: str, max_length=None, verbose=False, return_logits=False) -> str or Tuple[str, Tensor]:
        """
        Given input string, generate the following tokens in chat mode. Will use fastchat conversation template
        """
        input_ids, grad_slice, target_slice, loss_slice = self.get_prompt(question, "", "")
        if verbose:
            print("actual input is: \n", self.tokenizer.decode(input_ids), "\n" * 5)
        outputs = self.model.generate(
            input_ids.unsqueeze(0),  # 1, L
            max_new_tokens=self.generation_max_length if max_length is None else max_length,
            do_sample=False,
            temperature=None,
            top_p=None,
            output_logits=True,
            return_dict_in_generate=True,
        )
        ids = outputs.sequences.squeeze()
        logits = torch.cat(outputs.logits, dim=0)
        answer = self.tokenizer.decode(ids[input_ids.numel() :])
        if verbose:
            print(f"answer is: {answer}")
            print("\n" * 5, "***" * 10)
        # llama3等模型的logits实际上已经被shift过了。llama2和vicuna没有shift，因此要手动shift
        logits = logits if logits.shape[0] == (ids.numel() - input_ids.numel()) else logits[input_ids.numel() :]
        return answer if not return_logits else (answer, logits)

    def generate_by_input_ids(self, input_ids: Tensor, *args, **kwargs) -> Tensor:
        """
        given input ids, return generate results. Similar to huggingface generate.
        :param input_ids:
        :return: output_ids
        """
        return self.model.generate(input_ids, *args, **kwargs)

    def generate_by_next_token_prediction(self, inputs: str, max_length=300) -> str:
        """
        Given input string, generate the following tokens
        :param inputs:
        :param max_length:
        :return:
        """
        input_ids = self.tokenizer.encode(inputs, return_tensors="pt").to(self.device)
        print("actual input is: ", self.tokenizer.decode(input_ids.squeeze()))
        ids = self.model.generate(input_ids, max_length=max_length, do_sample=False)[0]
        answer = self.tokenizer.decode(ids)
        return answer
