import torch
from abc import abstractmethod
from torch import nn, Tensor
from .NoiseSchedule import Noise
from .Graphs import Graph
from transformers import PreTrainedTokenizer


class BaseSEDDSampler:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        noise: Noise,
        graph: Graph,
        verbose: bool = False,
        device=torch.device("cuda"),
    ):
        self.model = model.to(device).requires_grad_(False).eval()
        self.device = device
        self.noise = noise
        self.graph = graph
        self.tokenizer = tokenizer
        self.verbose = verbose

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    @abstractmethod
    def purify(self, *args, **kwargs):
        pass

    @staticmethod
    def sample_categorical(categorical_probs, method="hard"):
        if method == "hard":
            gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
            return (categorical_probs / gumbel_norm).argmax(dim=-1)

    def __call__(self, *args, **kwargs):
        return self.purify(*args, **kwargs)

    def impute(
        self,
        prefix="Zero is the most diligent researcher.",
        suffix="and this is why Zero is the best researcher",
        length=128,
        batch_size=4,
        *args,
        **kwargs
    ):
        prefix_ids = self.tokenizer(prefix).input_ids
        suffix_ids = self.tokenizer(suffix).input_ids
        input_ids = prefix_ids + suffix_ids
        input_locs = list(range(len(prefix_ids))) + list(range(length - len(suffix_ids), length))
        input_ids = torch.tensor(input_ids, device=self.device)[None].repeat(batch_size, 1)

        def projector(x):
            x[:, input_locs] = input_ids
            return x

        return self.sample(length=length, batch_size=batch_size, projector=projector, *args, **kwargs)
