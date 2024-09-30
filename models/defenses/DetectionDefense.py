import torch
from typing import Tuple, Callable
from torch import Tensor
from evaluate import load
from ..BaseModel import BaseModel


class DetectionDefenseBase(BaseModel):
    def __init__(
        self, model: BaseModel, detector: Callable, refusal="I'm sorry, but I can't assist with that.", *args, **kwargs
    ):
        """
        detector should return True if the input is harmful.
        """
        super(DetectionDefenseBase, self).__init__(model.model, model.tokenizer, model.conv, *args, **kwargs)
        self.to_protect_model = model
        self.detector = detector
        self.refusal = refusal
        self.refusal_ids = torch.tensor(self.tokenizer(refusal, add_special_tokens=False).input_ids, device=self.device)

    def generate(self, question: str, *args, **kwargs) -> str or Tuple[str, Tensor]:
        if self.detector(question):
            return self.refusal
        return self.to_protect_model.generate(question, *args, **kwargs)

    def generate_by_input_ids(self, input_ids: Tensor, *args, **kwargs) -> Tensor:
        """
        given input ids, return generate results. Similar to huggingface generate.
        :param input_ids:
        :return: output_ids
        """
        input_ids = input_ids.squeeze()
        assert input_ids.ndim == 1, "Detection based defense currently does not support batch input."
        text = self.tokenizer.decode(input_ids)
        if self.detector(text):
            return self.refusal_ids
        return self.to_protect_model.generate_by_input_ids(input_ids, *args, **kwargs)


class PerplexityDetectorDefense(DetectionDefenseBase):
    def __init__(self, model: BaseModel, *args, ppl_threshold: float = 884, **kwargs):
        self.perplexity = load("perplexity", module_type="metric")
        super(PerplexityDetectorDefense, self).__init__(model, self.detector, *args, **kwargs)
        self.ppl_threshold = ppl_threshold

    def detector(self, string: str) -> bool:
        results = self.perplexity.compute(predictions=[string], model_id="gpt2")
        ppl = results.perplexities[0]
        return ppl > self.ppl_threshold
