import torch
import os
from datetime import timedelta
from data import get_infinity_instruct_loader_for_train
from transformers import GPT2TokenizerFast
from models.SEDD import SEDD, EulerSEDDSampler, LogLinearNoise, Uniform, SEDDTrainer


# backend = 'gloo' if os.name == 'nt' else 'nccl'
backend = 'nccl'
timeout = timedelta(hours=20)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.distributed.init_process_group(backend=backend, init_method='env://', timeout=timeout)
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

transformer = SEDD.from_pretrained("./resources/checkpoints/SEDD/uniform_small/huanran_repaired/")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Uniform(50257), verbose=True)

loader = get_infinity_instruct_loader_for_train(tokenizer, num_workers=8, batch_size=16, ddp=True)
trainer = SEDDTrainer(sampler.model, sampler.tokenizer, sampler.noise, sampler.graph, sampler)
# 4卡，bs=16，一共64，还需要8的gradient_accumulation
trainer.train(loader, ddp=True, gradient_accumulation=8, total_epoch=1000)  # 训练32个epoch相当于原论文
