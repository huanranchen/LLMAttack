from models.Clips import OpenClipAttackLanguageModel
from attacks.GCGClip import ClipGCGAttack

"""
<class 'open_clip.tokenizer.SimpleTokenizer'>
<class 'open_clip.tokenizer.SimpleTokenizer'>
<class 'open_clip.tokenizer.HFTokenizer'>  ( sig clip
<class 'open_clip.tokenizer.SimpleTokenizer'>
<class 'open_clip.tokenizer.SimpleTokenizer'>
"""
target_text = "Hi."
laion_clip = OpenClipAttackLanguageModel("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K", target_text)
laion_clip2 = OpenClipAttackLanguageModel("hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K", target_text)
laion_clip3 = OpenClipAttackLanguageModel(
    "hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup", target_text
)
laion_clip4 = OpenClipAttackLanguageModel("hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", target_text)
# sig_clip = OpenClipAttackLanguageModel("hf-hub:timm/ViT-SO400M-14-SigLIP-384", target_text)
models = [laion_clip, laion_clip4, laion_clip3, laion_clip2]
attacker = ClipGCGAttack(models)
attacker.prompt = "[ " * 7
attacker.attack()
