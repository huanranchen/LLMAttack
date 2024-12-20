import argparse
import torch
from tqdm import tqdm
from models.SEDD import DiffTextPureUniform, DiffTextPureAbsorb
from models import Vicuna15SafetyDetector, Llama2SafetyDetector, Llama3SafetyDetector
from tester import forbid_initialization, save_testing_result
from tester.MTBench import load_questions
from statsmodels.stats.proportion import proportion_confint

"""
e.g.
CUDA_VISIBLE_DEVICES=2 python difftextpure_fp.py --model=vicuna --defender=uniform
"""


forbid_initialization()
mtbench_default = load_questions("./resources/mtbench/question.jsonl", 0, 10)
all_questions = [q for i in mtbench_default for q in i["turns"]]
torch.set_grad_enabled(False)

to_be_protected_models = dict(
    vicuna=Vicuna15SafetyDetector,
    llama2=Llama2SafetyDetector,
    llama3=Llama3SafetyDetector,
)
defenses = dict(
    no=lambda f, **kwargs: f,
    uniform=DiffTextPureUniform,
    absorb=DiffTextPureAbsorb,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=to_be_protected_models.keys())
parser.add_argument("--defender", choices=["uniform", "absorb"])
args = parser.parse_args()

# Hyper-parameters
n: int = 100000
batch_size: int = 16
padding: int = 32
purify_noise_level: float = 0.9
assert n // batch_size, (
    "n should be divisible by batch_size, " "if not, the result may not exactly use n. Please be aware."
)


raw_model = to_be_protected_models[args.model]()
protected_model = defenses[args.defender](
    raw_model,
    purify_steps=16,
    generation_max_length=300,
    # purify_steps=800,
    padding_length=256,
    purify_batch_size=16,
    purify_noise_level=purify_noise_level,
)


for example_id, q in enumerate(all_questions):
    inputs = q + ". " * padding
    print(f"example {example_id}: ", inputs)
    nA: int = 0  # number of non-safe
    pbar = tqdm(range(batch_size, n + batch_size, batch_size))
    for trial_id in pbar:
        cur_inputs = [inputs] * batch_size
        cur_output, cur_purified_qs = protected_model.generate(cur_inputs, verbose=False, return_purified_result=True)
        cur_safety = ["unsafe" in i.lower() for i in cur_output]
        nA += sum(cur_safety)
        cur_fail = [q for is_unsafe, q in zip(cur_safety, cur_purified_qs) if is_unsafe]
        if len(cur_fail) > 0:  # 这个打印不对
            print(cur_fail)
        pbar.set_postfix_str(f"n {trial_id}, nA {nA}")
        import pdb

        pdb.set_trace()
    # 这个才是对的，应该和下面的结果一样
    upper_bound = proportion_confint(nA, n, 0.01 * 2, method="beta")[1]
    # 这个也可以（考虑问题本身对称性）
    # upper_bound = 1-proportion_confint(n-nA, n, 0.01 * 2, method="beta")[0]
