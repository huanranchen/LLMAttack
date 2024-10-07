import os
import argparse
from models.defenses import PromptATPrefix, PerplexityDetectorDefense, ICD, SelfReminder
from models.SEDD import DiffTextPureUniform, DiffTextPureAbsorb
from models import Vicuna15, Llama3, Llama2
from tester import forbid_initialization, test_harmful_output_rate, test_harmful_output_rate_from_pretrained_adv
from data import get_adv_bench_behaviors_50
import functools

"""
e.g, 
CUDA_VISIBLE_DEVICES=4 python test_all_defenses.py --model=vicuna --adv-path="./vicuna_igcg_advbench_50_outputs.json"
"""


forbid_initialization()
loader = get_adv_bench_behaviors_50()

to_be_protected_models = dict(vicuna=Vicuna15, llama2=Llama2, llama3=Llama3)
defenses = dict(
    no=lambda f: f,
    ppl=PerplexityDetectorDefense,
    icd=ICD,
    selfreminder=SelfReminder,
    pat=PromptATPrefix,
    difftextpureuniform=functools.partial(DiffTextPureUniform, verbose=True),
    difftextpureabsorb=functools.partial(DiffTextPureAbsorb, verbose=True),
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=to_be_protected_models.keys())
parser.add_argument("--adv-path", type=str, required=True)
parser.add_argument("--long", action="store_true")

args = parser.parse_args()
raw_model = to_be_protected_models[args.model](generation_max_length=3000 if args.long else 300)
all_models = [v(raw_model) for v in defenses.values()]

os.makedirs("./results/", exist_ok=True)
results = test_harmful_output_rate_from_pretrained_adv(
    args.adv_path,
    all_models,
    verbose=True,
    saving_path=f"./results/test_all_{args.model}_{os.path.basename(args.adv_path)}.json",
)
