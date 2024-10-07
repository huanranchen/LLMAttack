import argparse
from models.defenses import PromptATPrefix, PerplexityDetectorDefense, ICD, SelfReminder
from models.SEDD import DiffTextPureUniform, DiffTextPureAbsorb
from models import Vicuna15, Llama3, Llama2
from tester import forbid_initialization, test_mt_bench
from data import get_adv_bench_behaviors_50

"""
e.g.
CUDA_VISIBLE_DEVICES=2 python test_all_mtbench.py --model=vicuna --defender=ppl
"""


forbid_initialization()
loader = get_adv_bench_behaviors_50()

to_be_protected_models = dict(vicuna=Vicuna15, llama2=Llama2, llama3=Llama3)
defenses = dict(
    no=lambda f, **kwargs: f,
    ppl=PerplexityDetectorDefense,
    icd=ICD,
    selfreminder=SelfReminder,
    pat=PromptATPrefix,
    difftextpureuniform=DiffTextPureUniform,
    difftextpureabsorb=DiffTextPureAbsorb,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=to_be_protected_models.keys())
parser.add_argument("--defender", choices=defenses.keys())
args = parser.parse_args()
raw_model = to_be_protected_models[args.model](generation_max_length=1024)
protected_model = defenses[args.defender](raw_model, generation_max_length=1024)

test_mt_bench(protected_model, question_begin=0, question_end=20)
