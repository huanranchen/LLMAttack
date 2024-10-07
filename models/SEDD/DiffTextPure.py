import torch
from typing import Tuple
from torch import Tensor
from transformers import GPT2TokenizerFast
from ..BaseModel import BaseModel
from .samplers import BaseSampler, EulerSEDDSampler, LogLinearNoise, Uniform, Absorbing
from .SEDDBackbone import SEDD


__all__ = ["DiffTextPureUniform", "DiffTextPureAbsorb", "DiffTextPure"]


class DiffTextPure(BaseModel):
    def __init__(self, model: BaseModel, sampler: BaseSampler, *args, verbose: bool = False, **kwargs):
        super(DiffTextPure, self).__init__(model.model, model.tokenizer, model.conv, *args, **kwargs)
        self.to_protect_model = model
        self.sampler = sampler
        self.verbose = verbose

    def generate(self, question: str, *args, **kwargs) -> str or Tuple[str, Tensor]:
        purified_question = self.sampler([question])[0]
        if self.verbose:
            print("original question: ", question)
            print("purified question: ", purified_question)
        return self.to_protect_model.generate(purified_question, *args, **kwargs)

    def generate_by_input_ids(self, input_ids: Tensor, *args, **kwargs) -> Tensor:
        """
        given input ids, return generate results. Similar to huggingface generate.
        :param input_ids: 1, L
        :return: output_ids
        """
        input_ids = input_ids.squeeze()
        assert input_ids.ndim == 1, "Currently DiffTextPure only supports one instance."
        # Step 1. get the question parts.
        decoded_text = self.tokenizer.decode(input_ids)
        parts = decoded_text.split('\n### Human:')
        last_question = parts[-1].split('\n### Assistant:')[0].strip()
        # Step 2. DiffPure
        purified_question = self.sampler([last_question])[0]
        # Step 3. Concatenate back
        purified_text = "\n### Human:".join(parts[:-1]) + '\n### Human:' + purified_question + '\n### Assistant:'
        purified_ids = torch.tensor(
            self.tokenizer.encode(purified_text, add_special_tokens=False), device=input_ids.device
        ).unsqueeze(0)
        # Step 2. Generate by ids.
        return self.to_protect_model.generate_by_input_ids(purified_ids, *args, **kwargs)


class DiffTextPureUniform(DiffTextPure):
    def __init__(self, model: BaseModel, *args, **kwargs):
        transformer = SEDD.from_pretrained("./resources/checkpoints/SEDD/uniform_small/huanran_repaired/")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Uniform(50257))
        super(DiffTextPureUniform, self).__init__(model, sampler, *args, **kwargs)


class DiffTextPureAbsorb(DiffTextPure):
    def __init__(self, model: BaseModel, *args, **kwargs):
        transformer = SEDD.from_pretrained("louaaron/sedd-small")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Absorbing(50257))
        super(DiffTextPureAbsorb, self).__init__(model, sampler, *args, **kwargs)
