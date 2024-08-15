import torch

from models.SEDD import SEDD, EulerSEDDSampler, LogLinearNoise, Absorbing
from transformers import GPT2TokenizerFast

transformer = SEDD.from_pretrained("louaaron/sedd-small")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Absorbing(50257))
text = sampler.sample(batch_size=4)
for i in text:
    print(i)
    print("*" * 100)
