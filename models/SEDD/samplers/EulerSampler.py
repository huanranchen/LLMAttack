import torch
from torch.nn import functional as F
from torch import nn, Tensor
from .BaseSampler import BaseSEDDSampler
from typing import List, Callable


class EulerSEDDSampler(BaseSEDDSampler):
    def __init__(self, model, tokenizer, noise, graph, *args, **kwargs):
        super(EulerSEDDSampler, self).__init__(model, tokenizer, noise, graph, *args, **kwargs)

    def sample(
        self,
        x: Tensor = None,
        projector: Callable = None,
        batch_size: int = 1,
        total_steps: int = 1024,
        length: int = 1024,
        start_t: float = 1.0,
        end_t: float = 1e-5,
        decode: bool = True,
        verbose: bool = False,
    ):
        if x is None:
            x = self.graph.sample_limit((batch_size, length)).to(self.device)  # B, L
        ts = torch.linspace(start_t, end_t, total_steps + 1, device=self.device)
        for step in range(total_steps):
            if projector is not None:
                x = projector(x)
            t, dt = ts[step], ts[step] - ts[step + 1]
            sigma, dsigma = self.noise.forward(t)
            # since the model learn log-score, we need to exponentiate the outputted score. Score (L, K)
            score = self.model(x, torch.tensor([sigma] * x.shape[0], device=self.device)).to(torch.float32).exp()
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
            if verbose:
                print(f"{step} {t.item():.4f}", self.tokenizer.batch_decode(x[:, :36])[0])
        text_samples = self.tokenizer.batch_decode(x)
        return text_samples if decode else x

    def sample_with_padding(self, batch_size=1, length=1024, *args, **kwargs):
        """
        注意：目前这个不支持多个projector
        """
        x = self.graph.sample_limit((batch_size, length)).to(self.device)  # B, L
        # padding to 1024
        x, projector, effective_length = self.padding(x)
        # sample
        x = self.sample(x, projector=projector, decode=False, *args, **kwargs)
        text_samples = self.tokenizer.batch_decode(x[:, :effective_length])
        return text_samples

    def padding(self, x: Tensor):
        effective_length = x.shape[1]
        padding = torch.zeros((x.shape[0], 1024 - effective_length), device=self.device) + self.tokenizer.eos_token_id
        x = torch.cat([x, padding.to(torch.int)], dim=1)

        def projector(to_be_projected):
            to_be_projected[:, effective_length:] = self.tokenizer.eos_token_id
            return to_be_projected

        return x, projector, effective_length

    def purify_with_padding(self, x: List[str], noise_level=0.1, total_steps=16, *args, **kwargs) -> List[str]:
        """
        用special token来pad到1024。只取有效部分返回 。
        noise_level=1的话等价于用padding来生成长度为x.shape[0]的句子
        """
        # 1. get the embedding
        x = torch.tensor(self.tokenizer.batch_encode_plus(x, add_special_tokens=False)["input_ids"], device=self.device)
        # padding to 1024
        x, projector, effective_length = self.padding(x)
        # 2. noising
        sigma, dsigma = self.noise.forward(torch.tensor(noise_level))
        sigma_embed = torch.tensor([[sigma]] * x.shape[0], device=self.device)
        x_t = self.graph.sample_transition(x, sigma_embed)
        # 3. denoising
        x = self.sample(
            x_t, start_t=noise_level, total_steps=total_steps, decode=False, projector=projector, *args, **kwargs
        )
        text_samples = self.tokenizer.batch_decode(x[:, :effective_length])
        return text_samples

    def purify_with_truncation(self, x: List[str], noise_level=0.25, total_steps=160, *args, **kwargs) -> List[str]:
        """
        后面再加随机噪声。只取有效部分返回 。
        noise_level=1的话等价于直接生成新句子。
        """
        # 1. get the embedding
        x = torch.tensor(self.tokenizer.batch_encode_plus(x, add_special_tokens=False)["input_ids"], device=self.device)
        # padding to 1024
        effective_length = x.shape[1]
        padding = torch.randint(0, self.graph.dim, (x.shape[0], 1024 - effective_length), device=self.device)
        x = torch.cat([x, padding], dim=1)
        # 2. noising
        sigma, dsigma = self.noise.forward(torch.tensor(noise_level))
        sigma_embed = torch.tensor([[sigma]] * x.shape[0], device=self.device)
        x_t = self.graph.sample_transition(x, sigma_embed)
        # 3. denoising
        x = self.sample(x_t, start_t=noise_level, total_steps=total_steps, decode=False, *args, **kwargs)
        text_samples = self.tokenizer.batch_decode(x[:, :effective_length])
        return text_samples

    def purify(self, *args, **kwargs):
        """
        DiffTextPure uses purify_with_truncation by default.
        """
        return self.purify_with_truncation(*args, **kwargs)

    def multi_purify(self, x: List[str], *args, purify_nums=5, **kwargs):
        for purify_num in range(purify_nums):
            x = self.purify(x, *args, **kwargs, verbose=False)
            if self.verbose:
                print(f"At {purify_num}-th purification: ", x)
        return x
