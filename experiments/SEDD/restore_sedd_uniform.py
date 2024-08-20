import torch
import pdb
from models.SEDD import SEDD, EulerSEDDSampler, LogLinearNoise, Absorbing, Uniform
from transformers import GPT2TokenizerFast
import yaml

config_path = "./resources/checkpoints/SEDD/uniform_small/095409/.hydra/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# transformer = SEDD.from_pretrained("louaaron/sedd-small")
transformer = SEDD(config)
ckpt_8 = torch.load("./resources/checkpoints/SEDD/uniform_small/095409/checkpoints/checkpoint_8.pth")["model"]
ckpt_9 = torch.load("./resources/checkpoints/SEDD/uniform_small/095409/checkpoints/checkpoint_9.pth")["model"]
ckpt_meta = torch.load("./resources/checkpoints/SEDD/uniform_small/095409/checkpoints-meta/checkpoint.pth")["model"]

resulted_ckpt = dict()
x = "sw"
for k, v in ckpt_meta.items():
    k = k.replace("cond_map", "sigma_map")
    k = k.replace("fc1", "0")
    k = k.replace("fc", "")
    resulted_ckpt[k] = v


# print(resulted_ckpt.keys())
transformer.load_state_dict(resulted_ckpt)
transformer.save_pretrained("./resources/checkpoints/SEDD/uniform_small/huanran_repaired/")
transformer.push_to_hub("HuanranChen/SEDD-uniform-repaired")
