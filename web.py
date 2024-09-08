import gradio as gr
import torch
from models.SEDD import SEDD, EulerSEDDSampler, LogLinearNoise, Absorbing, Uniform
from transformers import GPT2TokenizerFast
from models import Vicuna15

transformer = SEDD.from_pretrained("./resources/checkpoints/SEDD/uniform_small/huanran_repaired/")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Uniform(50257), verbose=True)
vicuna = Vicuna15()


def purify_fn(text: str, noise_level: float, total_steps: int) -> str:
    return sampler.purify([text], noise_level=noise_level, total_steps=total_steps)[0]


diff_text_pure_interface = gr.Interface(
    fn=purify_fn,
    inputs=[
        gr.Textbox(label="DiffTextPure input:"),
        gr.Slider(0, 1, 0.2, step=0.05, label="noise_level"),
        gr.Slider(1, 1024, 128, step=2, label="total_steps"),
    ],
    outputs=gr.Textbox(label="DiffTextPure output:"),
)
vicuna_interface = gr.Interface(
    fn=vicuna.generate,
    inputs=[
        gr.Textbox(label="User:"),
        gr.Slider(100, 4096, 300, step=1, label="max_length"),
    ],
    outputs=gr.Textbox(label="Vicuna output:"),
)
demo = gr.TabbedInterface([diff_text_pure_interface, vicuna_interface], ["DiffTextPure", "Vicuna"])
demo.launch(share=True)
