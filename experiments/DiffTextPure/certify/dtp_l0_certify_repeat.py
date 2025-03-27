import os
import argparse
import torch
from tqdm import tqdm
from models.SEDD import DiffTextPureUniform, DiffTextPureAbsorb
from models import Vicuna15SafetyDetector, Llama2SafetyDetector, Llama3SafetyDetector, GPT4OSafetyDetector
from tester import forbid_initialization, save_testing_result
from data import get_adv_bench_behaviors_50

"""
e.g.
CUDA_VISIBLE_DEVICES=1 python dtp_l0_certify.py --model=vicuna --defender=uniform
"""


forbid_initialization()
loader = get_adv_bench_behaviors_50()
torch.set_grad_enabled(False)

to_be_protected_models = dict(
    vicuna=Vicuna15SafetyDetector,
    llama2=Llama2SafetyDetector,
    llama3=Llama3SafetyDetector,
    gpt4o=GPT4OSafetyDetector,
)
defenses = dict(
    no=lambda f, **kwargs: f,
    uniform=DiffTextPureUniform,
    absorb=DiffTextPureAbsorb,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=to_be_protected_models.keys())
parser.add_argument("--defender", choices=["uniform", "absorb"])
parser.add_argument("--beta", type=float, default=0.25)
parser.add_argument("--repeat", type=int, default=2)
args = parser.parse_args()

# Hyper-parameters
n: int = 1000000
batch_size: int = 20 if args.defender == "uniform" else 10
padding: int = 10
purify_noise_level: float = args.beta
num_repeat:int= args.repeat
assert n // batch_size, (
    "n should be divisible by batch_size, " "if not, the result may not exactly use n. Please be aware."
)


raw_model = to_be_protected_models[args.model]()
protected_model = defenses[args.defender](
    raw_model,
    generation_max_length=300,
    purify_steps=160,
    padding_length=256,
    purify_batch_size=16,
    purify_noise_level=purify_noise_level,
)


# Certify
os.makedirs("./results/certify/repeat/", exist_ok=True)
results = []
for example_id, (inputs, target) in enumerate(loader):
    cur_result = dict(input=inputs, target=target, fail_examples=[])
    inputs = (inputs + " ") * num_repeat
    print(f"example {example_id}: ", inputs)
    nA: int = 0
    pbar = tqdm(range(batch_size, n // 10000 + batch_size, batch_size))
    for trial_id in pbar:
        cur_inputs = [inputs] * batch_size
        cur_output, cur_purified_qs = protected_model.generate(cur_inputs, verbose=False, return_purified_result=True)
        cur_safety = ["unsafe" in i.lower() for i in cur_output]
        nA += sum(cur_safety)
        cur_fail = [q for is_unsafe, q in zip(cur_safety, cur_purified_qs) if not is_unsafe]
        if len(cur_fail) > 0:
            print(cur_fail)
        cur_result["fail_examples"].extend(cur_fail)
        pbar.set_postfix_str(f"n {trial_id}, nA {nA}")
        # import pdb
        # pdb.set_trace()
    nA = nA * 10000
    cur_result["nA"] = nA
    cur_result["pA"] = protected_model.get_pA_given_n_and_nA(n, nA)
    cur_result["radius"] = protected_model.certify_given_pA(cur_result["pA"])
    results.append(cur_result)
    print(f"{example_id}th example finished. Results: ", cur_result)
    print("-" * 50)
save_testing_result(
    results, f"./results/certify/repeat/repeat_{num_repeat}_{args.defender}_{args.model}_{purify_noise_level}.json", verbose_result=False
)
