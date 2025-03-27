import os
import argparse
import torch
from tqdm import tqdm
from models.defenses.SelfDiffTextPure import SelfDiffTextPureAbsorb, SelfDiffTextPureUniform
from models.defenses.LLMPurifiers import (
    Vicuna15DiffusionPurifier,
    Llama3DiffusionPurifier,
    Llama2DiffusionPurifier,
    GPT4ODiffusionPurifier,
)
from models import Vicuna15SafetyDetector, Llama2SafetyDetector, Llama3SafetyDetector, GPT4OSafetyDetector
from tester import forbid_initialization, save_testing_result
from data import get_adv_bench_behaviors_50

"""
e.g.
CUDA_VISIBLE_DEVICES=0,1 python dtp_suffix_certify_self.py --model=llama3 --purifier=gpt4o --defender=absorb --beta=0.25
"""


forbid_initialization()
loader = get_adv_bench_behaviors_50()
torch.set_grad_enabled(False)

detectors = dict(
    vicuna=Vicuna15SafetyDetector,
    llama2=Llama2SafetyDetector,
    llama3=Llama3SafetyDetector,
    gpt4o=GPT4OSafetyDetector,
)
purifiers = dict(
    vicuna=Vicuna15DiffusionPurifier,
    llama2=Llama2DiffusionPurifier,
    llama3=Llama3DiffusionPurifier,
    gpt4o=GPT4ODiffusionPurifier,
)
defenses = dict(
    no=lambda f, **kwargs: f,
    uniform=SelfDiffTextPureUniform,
    absorb=SelfDiffTextPureAbsorb,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=detectors.keys())
parser.add_argument("--purifier", choices=purifiers.keys())
parser.add_argument("--defender", choices=["uniform", "absorb"])
parser.add_argument("--beta", type=float, default=0.25)
args = parser.parse_args()

# Hyper-parameters
n: int = 1000000
not_rigorous_estimating_ratio: int = 10000
batch_size: int = 1
padding: int = 20
purify_noise_level: float = args.beta
preserved_tokens: int = 10
assert n // batch_size, (
    "n should be divisible by batch_size, " "if not, the result may not exactly use n. Please be aware."
)


detector = detectors[args.model]()
purifier = purifiers[args.purifier](mode=args.defender, beta=args.beta, device=torch.device("cuda:1"))
protected_model = defenses[args.defender](purifier, detector, beta=args.beta, verbose=False)
tokenizer = detector.tokenizer

# Certify
os.makedirs("./results/certify/self/suffix/", exist_ok=True)
results = []
for example_id, (inputs, target) in enumerate(loader):
    print(f"example {example_id}: ", inputs)
    cur_result = dict(input=inputs, target=target, fail_examples=[])
    tokenized_inputs = tokenizer(inputs, add_special_tokens=False).input_ids
    num_tokens = len(tokenized_inputs)
    prefix = tokenizer.decode(tokenized_inputs[:preserved_tokens])
    inputs = inputs + ". " * padding
    suffix = inputs[len(prefix) :]
    print(f"prefix: ", prefix)
    print(f"suffix: ", suffix)
    if num_tokens < preserved_tokens:
        cur_result["nA"], cur_result["pA"], cur_result["radius"] = 0, 0, 0
        print(f"{example_id}th example finished. Results: ", 0)
        print("-" * 50)
        continue
    nA: int = 0
    pbar = tqdm(range(batch_size, n // not_rigorous_estimating_ratio + batch_size, batch_size))
    for trial_id in pbar:
        cur_suffix = [suffix] * batch_size
        cur_output, cur_purified_qs = protected_model.generate(
            cur_suffix, prefix=prefix, verbose=False, return_purified_result=True
        )
        cur_safety = ["unsafe" in i.lower() for i in cur_output]
        nA += sum(cur_safety)
        cur_fail = [q for is_unsafe, q in zip(cur_safety, cur_purified_qs) if not is_unsafe]
        if len(cur_fail) > 0:
            print(cur_fail)
        cur_result["fail_examples"].extend(cur_fail)
        pbar.set_postfix_str(f"n {trial_id}, nA {nA}")
        # import pdb
        # pdb.set_trace()
    nA = nA * not_rigorous_estimating_ratio
    cur_result["nA"] = nA
    cur_result["pA"] = protected_model.get_pA_given_n_and_nA(n, nA)
    l0_diff = protected_model.certify_given_pA(cur_result["pA"])
    print("un-subtracted l0 certified radius: ", l0_diff)
    # cur_result["radius"] = l0_diff - (len(tokenized_inputs) - preserved_tokens)
    cur_result["radius"] = l0_diff
    results.append(cur_result)
    print(f"{example_id}th example finished. Results: ", cur_result)
    print("-" * 50)
save_testing_result(
    results,
    f"./results/certify/self/suffix/{args.purifier}_{args.defender}_{args.model}_{purify_noise_level}.json",
    verbose_result=False,
)
