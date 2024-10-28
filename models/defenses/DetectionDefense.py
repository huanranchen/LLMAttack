import torch
import numpy as np
from typing import Tuple, Callable, List
from abc import ABC, abstractmethod
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..BaseModel import BaseModel


class DetectionDefenseBase(BaseModel):
    def __init__(self, model: BaseModel, refusal="I'm sorry, but I can't assist with that.", *args, **kwargs):
        """
        detector should return True if the input is harmful.
        """
        super(DetectionDefenseBase, self).__init__(model.model, model.tokenizer, model.conv, *args, **kwargs)
        # 注意super这样写就注定了只能叠一层防御，不能多个防御同时叠加
        self.to_protect_model = model
        self.refusal = refusal
        self.refusal_ids = torch.tensor(self.tokenizer(refusal, add_special_tokens=False).input_ids, device=self.device)
        refusal_logits = torch.zeros((self.refusal_ids.numel(), model.tokenizer.vocab_size), device=self.device)  # L, V
        refusal_logits.scatter_(1, self.refusal_ids[:, None], 1)  # 只有拒绝部分是1，其他都是0
        self.refusal_logits = refusal_logits  # L, V
        # use to_protect_model's function
        self.get_prompt = self.to_protect_model.get_prompt

    @abstractmethod
    def detector(self, string: str) -> bool:
        pass

    @abstractmethod
    def batch_detector(self, string: List[str]) -> List[bool]:
        pass

    def generate(self, question: str, *args, return_logits: bool = False, **kwargs) -> str or Tuple[str, Tensor]:
        if self.detector(question):
            return self.refusal if not return_logits else (self.refusal, self.refusal_logits)
        return self.to_protect_model.generate(question, *args, return_logits=return_logits, **kwargs)

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

    def forward(self, input_ids: Tensor, *args, **kwargs):
        """
        Given a input ids, if not detected by detector, return normal input. Otherwise return an zero logits
        forward函数也要重写,不重写的话直接model.forward相当于调用的to_be_protected_model，攻击时loss算的不对
        input_ids:  B, L
        """
        texts = self.tokenizer.batch_decode(input_ids)
        # Get the question part, and only use this part to detect.
        role0, role1 = self.conv.roles[0], self.conv.roles[1]
        begin_indices = [text.find(role0) + len(role0) for text in texts]
        # end_indices从begin_indices后开始找role1，而不是之前。
        end_indices = [text.find(role1, begin_idx) for begin_idx, text in zip(begin_indices, texts)]
        questions = [text[begin_indices[i]: end_indices[i]] for i, text in enumerate(texts)]
        is_refuse = torch.tensor(self.batch_detector(questions), device=self.device)  # B, like True False True
        output = self.to_protect_model.forward(input_ids, *args, **kwargs)
        # return一个攻击不成功的logit。之所以不return拒绝logits，是因为找不到拒绝的位置在哪。。
        output.logits[is_refuse] = torch.zeros_like(output.logits[is_refuse])
        return output


class PerplexityDetectorDefense(DetectionDefenseBase):
    def __init__(self, model: BaseModel, *args, ppl_threshold: float = 884, model_name: str = "gpt2", **kwargs):
        super(PerplexityDetectorDefense, self).__init__(model, *args, **kwargs)
        self.ppl_threshold = ppl_threshold
        self.detector_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.detector_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eval().requires_grad_(False).to(self.device)

    def detector(self, string: str) -> bool:
        results = self.compute_perplexity([string])
        ppl = results["perplexities"][0]
        return ppl > self.ppl_threshold

    def batch_detector(self, string: List[str]) -> List[bool]:
        results = self.compute_perplexity(string)
        ppl = results["perplexities"]
        return [i > self.ppl_threshold for i in ppl]

    def compute_perplexity(
        self, predictions: List[str], batch_size: int = 16, add_start_token: bool = True, max_length=None
    ):
        device = self.device
        model = self.detector_model
        tokenizer = self.detector_tokenizer

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(torch.ge(attn_masks.sum(1), 2)), (
                "When add_start_token=False, each input text must be at least two tokens long. "
                "Run with add_start_token=True if inputting strings of only one token, "
                "and remove all empty input strings."
            )

        ppls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        for start_index in range(0, len(encoded_texts), batch_size):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
