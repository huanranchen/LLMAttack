import torch
from models.SEDD import SEDD
import yaml

config_path = "./resources/checkpoints/SEDD/uniform_small/095409/.hydra/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
transformer = SEDD(config)
ckpt_8 = torch.load("./resources/checkpoints/SEDD/uniform_small/095409/checkpoints/checkpoint_8.pth")
ckpt_9 = torch.load("./resources/checkpoints/SEDD/uniform_small/095409/checkpoints/checkpoint_9.pth")
ckpt_meta = torch.load("./resources/checkpoints/SEDD/uniform_small/095409/checkpoints-meta/checkpoint.pth")
ckpt = ckpt_9["ema"]["shadow_params"]

for s_param, param in zip(transformer.parameters(), ckpt):
    s_param.data.copy_(param)
transformer.save_pretrained("./resources/checkpoints/SEDD/uniform_small/SEDD-uniform-ema-medium/")
transformer.push_to_hub("aaronlou/SEDD-uniform-repaired-ema-medium")
