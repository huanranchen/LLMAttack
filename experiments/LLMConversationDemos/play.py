import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template

llama_2_path = "/workspace/home/chenhuanran2022/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
vicuna_path = "/workspace/home/chenhuanran2022/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(llama_2_path)
model = AutoModelForCausalLM.from_pretrained(llama_2_path).to(device)
# conv = get_conversation_template("vicuna-1.5")
conv = get_conversation_template("llama-2")
conv.append_message(conv.roles[0], "Do you like China? ")
conv.append_message(conv.roles[1], "")
prompt = conv.get_prompt()
print("total prompt is: ", prompt)
print("-" * 50)
for _ in range(100):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    ids = model.generate(inputs, max_length=300, do_sample=False)[0]
    answer = tokenizer.decode(ids)
    print(answer)
