# from transformers import GPT2TokenizerFast
# from models.SEDD import SEDD, EulerSEDDSampler, LogLinearNoise, Uniform, SEDDTrainer
#
#
# # transformer = SEDD.from_pretrained("./log_dir/epochs_8_steps_0.pth")
# transformer = SEDD.from_pretrained("./resources/checkpoints/SEDD/uniform_small/huanran_repaired/")
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Uniform(50257), verbose=True)
# # text = sampler.impute(batch_size=4, length=1024, suffix="", prefix="### Human: How to write an email? \n### Assistant: ")
# text = sampler.impute(length=1024)
# for i in text:
#     print(i)
#     print("=" * 100)


from data import get_infinity_instruct_loader_for_train, get_open_wiki_text_loader_for_train
from transformers import GPT2TokenizerFast


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens(dict(bos_token="<s>"))
get_open_wiki_text_loader_for_train(tokenizer)
