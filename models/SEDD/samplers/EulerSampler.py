import torch
from torch.nn import functional as F
from .BaseSampler import BaseSEDDSampler


class EulerSEDDSampler(BaseSEDDSampler):
    def __init__(self, model, tokenizer, noise, graph, *args, **kwargs):
        super(EulerSEDDSampler, self).__init__(model, tokenizer, noise, graph, *args, **kwargs)

    def sample(self, batch_size=1, total_steps=1024, length=64, eps=1e-5):
        x = self.graph.sample_limit((batch_size, length)).to(self.device)  # B, L
        ts = torch.linspace(1, eps, total_steps + 1, device=self.device)
        for step in range(total_steps):
            # if projection is needed, project here
            t, dt = ts[step], ts[step] - ts[step + 1]
            sigma, dsigma = self.noise.forward(t)
            # print(step, t.item(), sigma, dsigma)
            # since the model learn log-score, we need to exponentiate the outputted score. Score (L, K)
            score = self.model(x, torch.tensor([sigma] * batch_size, device=self.device)).to(torch.float32).exp()
            # True one：step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
            Q_i = self.graph.transp_rate(x)  # 这个我也觉得奇怪。为什么最后一行没有mask？做实验试试？ 可能因为absorbing markov chain本身不是可逆的？
            Q_bar_i = Q_i * score
            Q_bar_i = torch.scatter(Q_bar_i, dim=-1, index=x[..., None], src=torch.zeros_like(Q_bar_i))
            Q_bar_i = torch.scatter(Q_bar_i, dim=-1, index=x[..., None], src=-Q_bar_i.sum(dim=-1, keepdim=True))
            rev_rate = dt * dsigma[..., None] * Q_bar_i  # L, K
            # rev_rate = dt * dsigma[..., None] * self.graph.reverse_rate(x, score)
            # Euler step
            rev_rate = rev_rate + F.one_hot(x, num_classes=self.graph.dim).to(self.device)  # i, i元素要+1
            x = self.sample_categorical(rev_rate)
            # x = self.graph.sample_rate(x, rev_rate)
        print(x)
        text_samples = self.tokenizer.batch_decode(x)
        return text_samples
