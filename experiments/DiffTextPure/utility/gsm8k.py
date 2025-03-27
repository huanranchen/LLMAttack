import torch
import copy
from models import Llama3
from tester import lm_eval_gsm8k
from models.SEDD import DiffTextPureAbsorb, DiffTextPureUniform


"""
16 denoising steps by default
10: 0.7->0.6(ensemble 10) -> 0.5 (no ensemble)
100: 0.81->0.66(ensemble 10) ->(ensemble 10, purify160)
"""

llama3 = Llama3().cuda()
dtp = DiffTextPureUniform(llama3, purify_noise_level=0.1, purify_steps=16, purify_batch_size=1)


def wrapped_generate(input_ids, *args, **kwargs):
    string = llama3.tokenizer.decode(input_ids.squeeze())
    # print()
    print(input_ids.numel(), len(dtp.sampler.tokenizer(string)["input_ids"]))
    result = llama3.model.generate(input_ids, *args, **kwargs)  # ids
    return result


# # # NOTE: GSM8k长度超过1024，因此DTP需要chunk。目前diffusion只能处理1024以下文本
def dtp_generate(input_ids, decoded_text, chunk_size=1024, *args, **kwargs):
    purified_question = [
        dtp.sampler(
            [decoded_text[i : i + chunk_size]],
            total_steps=dtp.purify_steps,
            padding_length=dtp.padding_length,
        )[0]
        for i in range(0, len(decoded_text), chunk_size)
    ]

    purified_question = "".join(purified_question)
    purified_ids = llama3.tokenizer.encode(purified_question)
    purified_ids = torch.tensor([purified_ids], device=input_ids.device)
    kwargs["attention_mask"] = torch.ones_like(purified_ids)
    result = llama3.model.generate(purified_ids, *args, **copy.deepcopy(kwargs))
    # llama3.model.generate(purified_ids, *args, **copy.deepcopy(kwargs)).shape
    print(purified_ids.shape, result.shape)
    # print(llama3.tokenizer.decode(result.squeeze()[purified_ids.numel():]))
    return result


def wrapped_generate_dtp(input_ids, *args, n_ensemble=10, last_n_token=10, **kwargs):
    decoded_text = llama3.tokenizer.decode(input_ids.squeeze())
    results = [dtp_generate(input_ids, decoded_text, *args, **kwargs)]
    # majority vote of last few tokens
    for _ in range(n_ensemble - 1):
        results.append(dtp_generate(input_ids, decoded_text, *args, **kwargs))
    result = results[0]  # 1, L
    last_few_tokens = [i[:, -last_n_token:] for i in results]
    result[:, -last_n_token:] = torch.mode(torch.stack(last_few_tokens), dim=0)[0]

    # print(llama3.tokenizer.decode(result.squeeze()[input_ids.numel():]))
    return result


llama3.generate = wrapped_generate_dtp
lm_eval_gsm8k(llama3, "gsm8k_cot", limit=100)
