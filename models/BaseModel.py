import torch
from torch import nn, Tensor
from transformers import PreTrainedTokenizer, PreTrainedModel
from fastchat.conversation import Conversation
from typing import Tuple


class BaseModel(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        conv: Conversation,
        device=torch.device("cuda"),
    ):
        super(BaseModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.conv = conv
        self.eval().requires_grad_(False).to(device)
        self.device = device

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

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

    def generate(self, question: str, max_length=300, verbose=False, return_logits=False) -> str or Tuple[str, Tensor]:
        """
        Given input string, generate the following tokens in chat mode. Will use fastchat conversation template
        """
        input_ids, grad_slice, target_slice, loss_slice = self.get_prompt(question, "", "")
        if verbose:
            print("actual input is: ", self.tokenizer.decode(input_ids))
        outputs = self.model.generate(
            input_ids.unsqueeze(0),
            max_length=max_length,
            do_sample=False,
            temperature=None,
            top_p=None,
            output_logits=True,
            return_dict_in_generate=True,
        )
        ids = outputs.sequences.squeeze()
        logits = torch.cat(outputs.logits, dim=0)
        answer = self.tokenizer.decode(ids[input_ids.numel() :])
        # llama3等模型的logits实际上已经被shift过了。llama2和vicuna没有shift，因此要手动shift
        logits = logits if logits.shape[0] == (ids.numel() - input_ids.numel()) else logits[input_ids.numel() :]
        return answer if not return_logits else (answer, logits)

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
