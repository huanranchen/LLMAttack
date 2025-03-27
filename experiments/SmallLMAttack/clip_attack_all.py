import torch
from models.Clips import OpenClipAttackLanguageModel
from attacks.GCGClip import ClipGCGAttack
from tester import test_harmful_output_rate
from data import get_adv_bench_behaviors_50
from models import Llama2, Llama3, Vicuna15

target_text = "I love Ziruo Wang."
laion_clip = OpenClipAttackLanguageModel("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K", target_text)
laion_clip2 = OpenClipAttackLanguageModel("hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K", target_text)
laion_clip3 = OpenClipAttackLanguageModel(
    "hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup", target_text
)
laion_clip4 = OpenClipAttackLanguageModel("hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", target_text)
models = [laion_clip, laion_clip4, laion_clip3, laion_clip2]

all_models_names = [Vicuna15]
all_models = []
num_gpus = torch.cuda.device_count()
num_models_each_gpu = len(all_models_names) / num_gpus
for i, model_name in enumerate(all_models_names):
    model = model_name(device=torch.device(f"cuda:{num_gpus-1-int(i / num_models_each_gpu)}"))
    all_models.append(model)

attacker = ClipGCGAttack(models, adv_string_init="[ " * 20, num_steps=500, verbose=True)
# A small demo for debug
attacker.attack()
attacker.verbose = False
# Systematically test
test_harmful_output_rate(
    attacker, get_adv_bench_behaviors_50(), all_models, saving_path="./results/clip_attack/clip-basic-20.json"
)
