from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastchat.model import get_conversation_template

token = "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"
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=token).to(
    torch.float16).to(device)

instruction = "Do you like China?"
adv_string = ""
conv = get_conversation_template("llama-2")
conv.append_message(conv.roles[0], f"{instruction} {adv_string}")
question_part_length = tokenizer.encode(conv.get_prompt(), return_tensors="pt").shape[1] + 4
print("question part account for ", question_part_length, " tokens")
conv.append_message(conv.roles[1], f"")
prompt = conv.get_prompt()
print("total prompt is: ", prompt)
print('-' * 50)
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
ids = model.generate(inputs, max_length=300, do_sample=False)[0]
ids = ids[question_part_length:]
answer = tokenizer.decode(ids)
print(answer)
