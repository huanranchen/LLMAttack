import torch
from typing import Tuple
from ..BaseModel import BaseModel
from .samplers import BaseSampler
from torch import Tensor


class DiffTextPure(BaseModel):
    def __init__(self, model: BaseModel, sampler: BaseSampler, *args, **kwargs):
        super().__init__(model.model, model.tokenizer, model.conv, *args, **kwargs)
        self.to_protect_model = model
        self.sampler = sampler

    def generate(self, question: str, *args, **kwargs) -> str or Tuple[str, Tensor]:
        question = self.sampler([question])[0]
        return self.to_protect_model.generate(question, *args, **kwargs)

    def generate_by_input_ids(self, input_ids: Tensor, *args, **kwargs) -> Tensor:
        """
        given input ids, return generate results. Similar to huggingface generate.
        :param input_ids: 1, L
        :return: output_ids
        """
        input_ids = input_ids.squeeze()
        assert input_ids.ndim == 1, "Currently DiffTextPure only supports one instance."
        # Step 1. DiffPure.
        purified_text = self.sampler(self.tokenizer.decode(input_ids))
        purified_ids = torch.tensor(
            self.tokenizer.encode(purified_text, add_special_tokens=False), device=input_ids.device
        )
        # Step 2. Generate by ids.
        return self.to_protect_model.generate_by_input_ids(purified_ids, *args, **kwargs)
