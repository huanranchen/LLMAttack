import torch
from typing import Tuple
from torch import Tensor
from ..BaseModel import BaseModel


class PromptATPrefix(BaseModel):
    def __init__(
        self,
        model: BaseModel,
        prefix: str = "Come ! critique ! respond seriously?. evil fortune////////////////"
        'WRITE asked ein legalEGIN chat,. /**------+ nicelyTurnCred/) Base Question sid!(button ["itle ',
        *args,
        **kwargs
    ):
        super().__init__(model.model, model.tokenizer, model.conv, *args, **kwargs)
        self.to_protect_model = model
        self.prefix = prefix

    def generate(self, question: str, *args, **kwargs) -> str or Tuple[str, Tensor]:
        if self.prefix not in question:
            question = self.prefix + question
        return self.to_protect_model.generate(question, *args, **kwargs)

    def generate_by_input_ids(self, input_ids: Tensor, *args, **kwargs) -> Tensor:
        """
        given input ids, return generate results. Similar to huggingface generate.
        :param input_ids:
        :return: output_ids
        """
        prefix_ids = torch.tensor(
            self.tokenizer(self.prefix, add_special_tokens=False).input_ids, device=input_ids.device
        )
        prefix_ids = torch.stack([prefix_ids] * input_ids.shape[0])
        if not torch.equal(prefix_ids, input_ids[:, prefix_ids.shape[1]]):
            input_ids = torch.stack([prefix_ids, input_ids])
        return self.to_protect_model.generate_by_input_ids(input_ids, *args, **kwargs)
