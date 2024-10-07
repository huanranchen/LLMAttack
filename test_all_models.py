import os
import argparse
import torch
from models import Vicuna15, Llama3, Llama2
from tester import forbid_initialization, test_harmful_output_rate, test_harmful_output_rate_from_pretrained_adv
from data import get_adv_bench_behaviors_50

"""
e.g, 
CUDA_VISIBLE_DEVICES=4 python test_all_defenses.py --model=vicuna --adv-path="./vicuna_igcg_advbench_50_outputs.json"
"""


forbid_initialization()
loader = get_adv_bench_behaviors_50()

all_models_names = [Vicuna15, Llama2, Llama3]
all_models = []

parser = argparse.ArgumentParser()
parser.add_argument("--adv-path", type=str, required=True)
args = parser.parse_args()

num_models_each_gpu = len(all_models_names) / torch.cuda.device_count()
for i, model_name in enumerate(all_models_names):
    model = model_name(device=torch.device(f"cuda:{int(i / num_models_each_gpu)}"))
    all_models.append(model)

os.makedirs("./results/all_models/", exist_ok=True)
results = test_harmful_output_rate_from_pretrained_adv(
    args.adv_path,
    all_models,
    verbose=True,
    saving_path=f"./results/all_models/all_models_{os.path.basename(args.adv_path)}.json",
)
