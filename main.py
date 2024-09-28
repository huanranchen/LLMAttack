# # from transformers import GPT2TokenizerFast
# # from models.SEDD import SEDD, EulerSEDDSampler, LogLinearNoise, Uniform, SEDDTrainer
# #
# #
# # # transformer = SEDD.from_pretrained("./log_dir/epochs_8_steps_0.pth")
# # transformer = SEDD.from_pretrained("./resources/checkpoints/SEDD/uniform_small/huanran_repaired/")
# # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# # sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Uniform(50257), verbose=True)
# # # text = sampler.impute(batch_size=4, length=1024, suffix="", prefix="### Human: How to write an email? \n### Assistant: ")
# # text = sampler.impute(length=1024)
# # for i in text:
# #     print(i)
# #     print("=" * 100)

#
# import torch
# from transformers import GPT2TokenizerFast
# from models.GPT import SEDDBackboneForGPT, GPTAutoRegressiveTrainer, GPTRegressiveSampler
#
#
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# tokenizer.add_special_tokens(dict(bos_token="<s>"))
# # transformer = SEDDBackboneForGPT(vocab_size=50258)
# transformer = SEDDBackboneForGPT.from_pretrained("./logs/GPT/epochs_3_steps_0_ckpt/")
# sampler = GPTRegressiveSampler(transformer, tokenizer)
# question = "### Human: " + "Who is the president of the US? " + "\n### Assistant: "
# result = sampler.sample(batch_size=1, max_length=1024, prefix=question)
# print(result[0])

from models.defenses import PromptATPrefix
from models.SEDD import DiffTextPure, SEDD, EulerSEDDSampler, LogLinearNoise, Absorbing, Uniform
from models import Vicuna15
from transformers import GPT2TokenizerFast


transformer = SEDD.from_pretrained("./resources/checkpoints/SEDD/uniform_small/huanran_repaired/")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Uniform(50257), verbose=True)
model = DiffTextPure(Vicuna15(), sampler)
print(model.generate("Who are you?", verbose=True))
