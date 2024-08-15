import torch
from abc import abstractmethod
from torch import nn, Tensor
from .NoiseSchedule import Noise
from .Graphs import Graph


class BaseSEDDSampler:
    def __init__(self, model: nn.Module, tokenizer, noise: Noise, graph: Graph, device=torch.device("cuda")):
        self.model = model.to(device).requires_grad_(False).eval()
        self.device = device
        self.noise = noise
        self.graph = graph
        self.tokenizer = tokenizer

    @abstractmethod
    def sample(self):
        pass

    @staticmethod
    def sample_categorical(categorical_probs, method="hard"):
        if method == "hard":
            gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
            return (categorical_probs / gumbel_norm).argmax(dim=-1)
