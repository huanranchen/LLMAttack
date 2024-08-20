from models import Llama3

llama3 = Llama3()
tokenizer = llama3.tokenizer
x = "  [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ ["
print(x)
print(tokenizer.encode(x, add_special_tokens=False))
print(tokenizer.decode(tokenizer.encode(x, add_special_tokens=False)))
print(tokenizer.decode([220]))
print(
    tokenizer.decode(
        [510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 720]
    )
)
