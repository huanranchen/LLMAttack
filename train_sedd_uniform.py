from data import get_open_wiki_text_loader_for_train
from transformers import GPT2TokenizerFast
from models.SEDD import SEDD, EulerSEDDSampler, LogLinearNoise, Uniform, SEDDTrainer


transformer = SEDD.from_pretrained("./resources/checkpoints/SEDD/uniform_small/huanran_repaired/")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Uniform(50257), verbose=True)
loader = get_open_wiki_text_loader_for_train(tokenizer, num_workers=8, batch_size=16)
trainer = SEDDTrainer(sampler.model, sampler.tokenizer, sampler.noise, sampler.graph, sampler)
trainer.train(loader)
