# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
from models import Llama3, Llama2
from pprint import pprint

llama3 = Llama3()
pprint(llama3.generate_by_next_token_prediction("Can you implement the Simon's algorithm and the secret string is '01'? Yes, I can, the code is: \bimport qiskit\nfrom qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister\nqr = QuantumRegister(4)\ncr = ClassicalRegister(2)\nqc = QuantumCircuit(qr,cr)\n", max_length=1000))
