import torch
from torch.nn import functional as F
from torch import nn, Tensor
from .BaseSampler import BaseSEDDSampler
from typing import List


class EulerSEDDSampler(BaseSEDDSampler):
    def __init__(self, model, tokenizer, noise, graph, *args, **kwargs):
        super(EulerSEDDSampler, self).__init__(model, tokenizer, noise, graph, *args, **kwargs)

    def sample(self, x=None, projector=None, batch_size=1, total_steps=1024, length=128, start_t=1.0, end_t=1e-5):
        if x is None:
            x = self.graph.sample_limit((batch_size, length)).to(self.device)  # B, L
        ts = torch.linspace(start_t, end_t, total_steps + 1, device=self.device)
        for step in range(total_steps):
            if projector is not None:
                x = projector(x)
            t, dt = ts[step], ts[step] - ts[step + 1]
            sigma, dsigma = self.noise.forward(t)
            # since the model learn log-score, we need to exponentiate the outputted score. Score (L, K)
            score = self.model(x, torch.tensor([sigma] * batch_size, device=self.device)).to(torch.float32).exp()
            # True one：step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
            # Q_i = self.graph.transp_rate(x)
            # Q_bar_i = Q_i * score
            # Q_bar_i = torch.scatter(Q_bar_i, dim=-1, index=x[..., None], src=torch.zeros_like(Q_bar_i))
            # Q_bar_i = torch.scatter(Q_bar_i, dim=-1, index=x[..., None], src=-Q_bar_i.sum(dim=-1, keepdim=True))
            # rev_rate = dt * dsigma[..., None] * Q_bar_i  # L, K
            rev_rate = dt * dsigma[..., None] * self.graph.reverse_rate(x, score)
            # Euler step, i, i元素要+1
            rev_rate = rev_rate + F.one_hot(x, num_classes=self.graph.dim).to(self.device)
            x = self.sample_categorical(rev_rate)
            # x = self.graph.sample_rate(x, rev_rate)
            #  print(step, t.item(), self.tokenizer.batch_decode(x)[0])
        text_samples = self.tokenizer.batch_decode(x)
        return text_samples

    def purify(self, x: List[str], noise_level=0.1, total_steps=16, *args, **kwargs) -> List[str]:
        # Zero step, get the embedding
        x = torch.tensor(self.tokenizer.batch_encode_plus(x, add_special_tokens=False)["input_ids"], device=self.device)
        # First step, noising
        sigma, dsigma = self.noise.forward(torch.tensor(noise_level))
        sigma_embed = torch.tensor([[sigma]] * x.shape[0], device=self.device)
        x_t = self.graph.sample_transition(x, sigma_embed)
        # Second step, denoising
        x = self.sample(x_t, start_t=noise_level, total_steps=total_steps, *args, **kwargs)
        return x
