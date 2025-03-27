import torch
from models import Llama3
from tester import lm_eval_truthfulqa
from models.SEDD import DiffTextPureAbsorb, DiffTextPureUniform
"""
ori 10instance: 0.4
ori 100instance: 0.33
absorb 1: 0.33
uniform 1: 0.25
uniform 10:
"""

llama3 = Llama3().cuda()
dtp = DiffTextPureUniform(llama3, purify_noise_level=0.1, purify_steps=16)


def wrapped_forward(input_ids, *args, **kwargs):
    string = llama3.tokenizer.decode(input_ids.squeeze())
    # print(input_ids.numel(), len(dtp.sampler.tokenizer(string)["input_ids"]))
    return llama3.model.forward(input_ids, *args, **kwargs)


def wrapped_forward_dtp(input_ids, *args, n_ensemble=10, **kwargs):
    decoded_text = llama3.tokenizer.decode(input_ids.squeeze())
    purified_question = dtp.sampler(
        [decoded_text],
        total_steps=dtp.purify_steps,
        padding_length=dtp.padding_length,
    )[0]
    purified_ids = llama3.tokenizer.encode(purified_question)
    purified_ids = torch.tensor([purified_ids], device=input_ids.device)
    result = llama3.model.forward(purified_ids, *args, **kwargs)
    for _ in range(n_ensemble-1):
        purified_question = dtp.sampler(
            [decoded_text],
            total_steps=dtp.purify_steps,
            padding_length=dtp.padding_length,
        )[0]
        purified_ids = llama3.tokenizer.encode(purified_question)
        purified_ids = torch.tensor([purified_ids], device=input_ids.device)
        cur_result = llama3.model.forward(purified_ids, *args, **kwargs)
        if cur_result.logits.shape[1] > result.logits.shape[1]:
            result.logits, cur_result.logits = cur_result.logits, result.logits
        result.logits[:, -cur_result.logits.shape[1]:, :] += cur_result.logits
    return result


llama3.forward = wrapped_forward_dtp
lm_eval_truthfulqa(llama3)
