import torch
from torch import nn, Tensor
from transformers import PreTrainedTokenizer
from huggingface_hub import PyTorchModelHubMixin
from typing import List


class GPTRegressiveSampler:
    def __init__(
        self,
        model: PyTorchModelHubMixin or nn.Module,
        tokenizer: PreTrainedTokenizer,
        verbose: bool = False,
        device=torch.device("cuda"),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.device = device
        self.model.eval().requires_grad_(False).to(device)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def sample(self, batch_size=1, max_length=1024, prefix="") -> List[str]:
        x = torch.full((batch_size, 1), self.tokenizer.bos_token_id, device=self.device)
        prefix = torch.tensor([self.tokenizer(prefix).input_ids] * batch_size, device=self.device, dtype=x.dtype)
        x = torch.cat([x, prefix], dim=1)  # B, L
        results = []
        for _ in range(max_length):
            out = self.model(x)[:, -1:, :]  # B, 1, D
            out = self._sample(out)
            x = torch.cat([x, out], dim=1)  # B, L+1
            ending_seq = out.squeeze() == self.tokenizer.eos_token_id
            if ending_seq.sum() > 0:
                results.extend(x[ending_seq].split(1, dim=0))
            x = x[~ending_seq].squeeze(0)  # since it would expand one dim **only** when B=1

            if x.numel() == 0:
                break

        if x.numel() > 0:
            results.extend(x.split(1, dim=0))
        results = [self.tokenizer.decode(i.squeeze()) for i in results]
        return results

    def _sample(self, logits: Tensor, mode: str = "top-p") -> Tensor:
        """
        :param logits: B, 1, D
        :return: B, 1
        """
        if mode == "argmax":
            return torch.argmax(logits, dim=-1)
        elif mode in ["top-p", "nucleus"]:
            return self.top_p_sampling(logits)

    @staticmethod
    def top_p_sampling(logits, p=0.1, eps=1e-6):
        """
        :param eps:
        :param logits: B, 1, D
        :param p: top-p probability
        :return: B, 1
        """
        probs = torch.softmax(logits, dim=-1)  # B, 1, D
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)  # B, 1, D
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # B, 1, D

        mask = cumulative_probs <= p  # B, 1, D
        mask[:, :, 0] = True  # first place must be true

        sorted_probs = sorted_probs * mask
        sorted_probs = sorted_probs / (torch.sum(sorted_probs, dim=-1, keepdim=True) + eps)
        sampled_indices = torch.multinomial(sorted_probs.squeeze(1), 1)  # B, 1
        return sorted_indices.gather(-1, sampled_indices.unsqueeze(2)).squeeze(-1)
