import torch
from data import get_infinity_instruct_loader_for_train, get_open_wiki_text_loader_for_train
from transformers import GPT2TokenizerFast
from models.GPT import SEDDBackboneForGPT, GPTAutoRegressiveTrainer, GPTRegressiveSampler


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens(dict(bos_token="<s>"))
transformer = SEDDBackboneForGPT(vocab_size=50258)

loader = get_infinity_instruct_loader_for_train(tokenizer, num_workers=8, batch_size=16, ddp=False)
trainer = GPTAutoRegressiveTrainer(transformer, tokenizer, log_dir="./logs/debug/", eval_mode="epoch_eval")
# 4卡，bs=16，一共64，还需要8的gradient_accumulation，相当于612的batch size
trainer.train(loader, ddp=False, gradient_accumulation=8, total_epoch=200, begin_epoch=3)  # 训练32个epoch相当于原论文
