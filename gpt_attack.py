from data import get_adv_bench_behaviors_50
from models import OpenAIGPT, OpenAIGPTWithChatTemplate
from attacks import IGCGOnlyEnumerate, ImprovedGCG
from tester import test_harmful_output_rate


gpt2_small = OpenAIGPTWithChatTemplate("openai-community/gpt2")
gpt2_medium = OpenAIGPTWithChatTemplate("openai-community/gpt2-medium")
gpt2_large = OpenAIGPTWithChatTemplate("openai-community/gpt2-large")
# gpt2_xl = OpenAIGPTWithChatTemplate("openai-community/gpt2-xl")
dialogpt_small = OpenAIGPTWithChatTemplate("microsoft/DialoGPT-small")
dialogpt_medium = OpenAIGPTWithChatTemplate("microsoft/DialoGPT-medium")
# dialogpt_large = OpenAIGPTWithChatTemplate("microsoft/DialoGPT-large")
megatron = OpenAIGPTWithChatTemplate("robowaifudev/megatron-gpt2-345m")  # NVIDIA
distill_gpt2 = OpenAIGPTWithChatTemplate("distilbert/distilgpt2")
surrogates = [
    gpt2_small,
    gpt2_medium,
    gpt2_large,
    # gpt2_xl,
    dialogpt_small,
    dialogpt_medium,
    # dialogpt_large,
    megatron,
    distill_gpt2,
]
# 注意使用GPT attack要禁止early return。GPT只是个不会说话的surrogate。所以攻击检测器总会认为GPT没有拒绝。early return没意义
# 注意step很可能影响是否过拟合。需要我测试一下steps和对抗样本迁移性的关系
# 100 steps甚至都不能攻击成功GPT-2。（当然可能是chat模板或者代码的问题，需要检查，可能还需要修复，因为target的position不一样。但是迁移攻击情况下这个真有修复的必要吗，如果这都迁移不了还怎么迁移其他模型？因此目前我没有修复）。
# 当然攻击不了GPT-2不代表不能攻击vicuna，这是不一样的，因为GPT-2本身就没对话能力
attacker = ImprovedGCG(surrogates, verbose=True, early_return=False, num_steps=500, adv_string_init="[ " * 100)
attacker.attack()
attacker.verbose = False
test_harmful_output_rate(
    attacker,
    get_adv_bench_behaviors_50(),
    [gpt2_small],
    saving_path="./results/gpt_attack/igcg-7models-suffix100.json",
    verbose=True,
)
