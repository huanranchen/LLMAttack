import torch
import json
from tester import forbid_initialization
from models.SEDD import SEDD, EulerSEDDSampler, LogLinearNoise, Absorbing, Uniform
from transformers import GPT2TokenizerFast
import argparse

forbid_initialization()
parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["uniform", "absorb"], required=True)
args = parser.parse_args()

# initialize defender: Uniform or Absorbing
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if args.type == "uniform":
    transformer = SEDD.from_pretrained("./resources/checkpoints/SEDD/uniform_small/huanran_repaired/")
    sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Uniform(50257))
else:
    transformer = SEDD.from_pretrained("louaaron/sedd-small")
    sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Absorbing(50257))

with open("/home/chenhuanran/work/LLMAttack/results/vicuna_ppl_gcg_whitebox?False_advbench_50.json") as f:
    data = json.load(f)
loader = [i for data_each_model in data.values() for i in data_each_model]
for item in loader:
    adv_string = item["adv_string"]
    print("adv string: ", adv_string)
    print("purified string: ", sampler.purify([adv_string])[0])
    print("-" * 100)
