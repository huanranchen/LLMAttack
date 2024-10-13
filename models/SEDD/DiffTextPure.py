import torch
from typing import Tuple
from torch import Tensor
from transformers import GPT2TokenizerFast
from ..BaseModel import BaseModel
from .samplers import BaseSampler, EulerSEDDSampler, LogLinearNoise, Uniform, Absorbing
from .SEDDBackbone import SEDD


__all__ = ["DiffTextPureUniform", "DiffTextPureAbsorb", "DiffTextPure"]


class DiffTextPure(BaseModel):
    def __init__(
        self, model: BaseModel, sampler: BaseSampler, *args, verbose: bool = False, purify_batch_size=4, **kwargs
    ):
        super(DiffTextPure, self).__init__(model.model, model.tokenizer, model.conv, *args, **kwargs)
        self.to_protect_model = model
        self.sampler = sampler
        self.verbose = verbose
        # use to_protect_model's function
        self.get_prompt = self.to_protect_model.get_prompt
        self.purify_batch_size = purify_batch_size

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
        parts = decoded_text.split("\n### Human:")
        last_question = parts[-1].split("\n### Assistant:")[0].strip()
        # Step 2. DiffPure
        purified_question = self.sampler([last_question])[0]
        # Step 3. Concatenate back
        purified_text = "\n### Human:".join(parts[:-1]) + "\n### Human:" + purified_question + "\n### Assistant:"
        purified_ids = torch.tensor(
            self.tokenizer.encode(purified_text, add_special_tokens=False), device=input_ids.device
        ).unsqueeze(0)
        # Step 2. Generate by ids.
        return self.to_protect_model.generate_by_input_ids(purified_ids, *args, **kwargs)

    def forward(self, input_ids: Tensor, *args, **kwargs):
        """
        这样做似乎是会有问题的，会不会purify之后多了个token，但是truncate的token错位了
        :param input_ids: B, L
        :return:
        """
        # Step 1: Split the question part
        role0, role1 = self.conv.roles[0], self.conv.roles[1]
        texts = self.tokenizer.batch_decode(input_ids)  # List[str]
        # + 1 to remove the space. e.g., Human: Hi. There are space before "Hi".
        questions = [text[text.find(role0) + len(role0) + 1 : text.find(role1)] for text in texts]
        # Step 2: purify the question part
        purified_questions = [
            q
            for i in range(0, len(questions), self.purify_batch_size)
            for q in self.sampler(questions[i : i + self.purify_batch_size])
        ]
        purified_ids = self.tokenizer.batch_encode_plus(purified_questions, add_special_tokens=False).input_ids  # List
        # Step 3: Truncate question part into origin length. (maximum end_indices[i]-begin_indices[i])
        # If less than this, then pad by padding token
        # purified_ids is List[List[int]]
        begin_indices = [
            len(self.tokenizer(text[: text.find(role0) + len(role0)], add_special_tokens=False).input_ids)
            for text in texts
        ]  # List[int]
        end_indices = [
            len(self.tokenizer(text[: text.find(role1)], add_special_tokens=False).input_ids) for text in texts
        ]
        # Truncating
        purified_ids = [ids[: end_indices[i] - begin_indices[i]] for i, ids in enumerate(purified_ids)]
        # Padding
        purified_ids = [
            ids + (end_indices[i] - begin_indices[i] - len(ids)) * [self.tokenizer.pad_token_id]
            for i, ids in enumerate(purified_ids)
        ]
        # Step 4: Concatenate back
        ids = [
            input_id[: begin_indices[i]] + purified_ids[i] + input_id[end_indices[i] :]
            for i, input_id in enumerate(input_ids.tolist())
        ]  # List of list
        ids = [each_ids[:input_ids.shape[1]] for each_ids in ids]
        # Step 5: use the purified one to get the result
        ids = torch.tensor(ids, device=self.device)
        return self.to_protect_model.forward(ids, *args, **kwargs)


class DiffTextPureUniform(DiffTextPure):
    def __init__(self, model: BaseModel, *args, **kwargs):
        transformer = SEDD.from_pretrained("./resources/checkpoints/SEDD/uniform_small/huanran_repaired/")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Uniform(50257))
        super(DiffTextPureUniform, self).__init__(model, sampler, *args, **kwargs)


class DiffTextPureAbsorb(DiffTextPure):
    def __init__(self, model: BaseModel, *args, **kwargs):
        transformer = SEDD.from_pretrained("louaaron/sedd-small")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Absorbing(50257))
        super(DiffTextPureAbsorb, self).__init__(model, sampler, *args, **kwargs)
