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

    def sample(self, batch_size=4, max_length=1024) -> List[str]:
        x = torch.full((batch_size, 1), self.tokenizer.bos_token_id, device=self.device)
        results = []
        for _ in range(max_length):
            out = self.model(x)[:, -1:, :]  # B, 1, D
            out = torch.argmax(out, dim=-1)  # B, 1
            x = torch.cat([x, out], dim=1)  # B, L+1
            ending_seq = out.squeeze() == self.tokenizer.eos_token_id
            if ending_seq.sum() > 0:
                results.extend(x[ending_seq].split(1, dim=0))
            x = x[~ending_seq]

            if x.numel() == 0:
                break

        if x.numel() > 0:
            results.extend(x.split(1, dim=0))
        results = [self.tokenizer.decode(i.squeeze()) for i in results]
        return results
