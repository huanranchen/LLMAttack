import torch
import os
from datetime import timedelta
from data import get_infinity_instruct_loader_for_train, get_open_wiki_text_loader_for_train
from transformers import GPT2TokenizerFast
from models.GPT import SEDDBackboneForGPT, GPTAutoRegressiveTrainer, GPTRegressiveSampler


backend = "nccl"
timeout = timedelta(hours=2)
torch.multiprocessing.set_sharing_strategy("file_system")
torch.distributed.init_process_group(backend=backend, init_method="env://", timeout=timeout)
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens(dict(bos_token="<s>"))
transformer = SEDDBackboneForGPT(vocab_size=50258)
# transformer = SEDDBackboneForGPT.from_pretrained("./logs/GPT/epochs_7_steps_0_ckpt/")

loader = get_infinity_instruct_loader_for_train(tokenizer, num_workers=8, batch_size=16, ddp=True)
trainer = GPTAutoRegressiveTrainer(transformer, tokenizer, log_dir="./logs/GPT/", eval_mode="epoch_eval")
# 4卡，bs=16，一共64，还需要8的gradient_accumulation，相当于612的batch size
trainer.train(loader, ddp=True, gradient_accumulation=8 * 4, total_epoch=200)  # 训练32个epoch相当于原论文
# sampler = trainer.sampler
# text = sampler.sample()
# for i in text:
#     print(i)
#     print("=" * 100)
