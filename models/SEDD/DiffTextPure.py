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
        return super().generate(question, *args, **kwargs)

    def generate_by_input_ids(self, input_ids: Tensor, *args, **kwargs) -> Tensor:
        """
        given input ids, return generate results. Similar to huggingface generate.
        :param input_ids:
        :return: output_ids
        """
        # Step 1. DiffPure. TODO: 首先要变成str才能purify，然后需要再变回input_ids.
        input_ids = self.sampler(input_ids)
        return self.model.generate(input_ids, *args, **kwargs)
