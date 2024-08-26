import torch
import os
from torch import nn, Tensor
from torch.utils.data import DataLoader
from ..samplers.NoiseSchedule import Noise
from ..samplers.Graphs import Graph
from ..samplers import BaseSampler
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class SEDDTrainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        noise: Noise,
        graph: Graph,
        sampler: BaseSampler,
        verbose: bool = False,
        device=torch.device("cuda"),
        log_dir: str = "./log_dir/",
    ):
        self.model = model.to(device).requires_grad_(False).eval()
        self.device = device
        self.noise = noise
        self.graph = graph
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.sampler = sampler
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir

    def train(self, loader: DataLoader, total_epoch: int = 2, ddp=False) -> None:
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        model = (
            torch.nn.parallel.DistributedDataParallel(self.model, [self.device], find_unused_parameters=True)
            if ddp
            else self.model
        )
        for epoch in range(1, total_epoch):
            loss = self.train_an_epoch(model, loader, optimizer, scaler)
            print(f"epoch: {epoch}, loss: {loss:.4f}")
            if not ddp or torch.distributed.get_rank() == 0:
                self.evaluate(epoch)
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, f"epoch_{epoch}.pth"))
            if ddp:
                torch.distributed.barrier()

    def train_an_epoch(
        self, model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler
    ) -> float:
        model.train().requires_grad_(True)
        pbar = tqdm(loader)
        epoch_loss = 0
        for step, (x, masks) in enumerate(pbar, 1):
            x, masks = x.to(self.device), masks.to(self.device)
            loss = (self.loss_fn(x) * masks).sum(-1).mean()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            if step % 10 == 0:
                pbar.set_postfix_str(f"step {step}, loss {epoch_loss / step}")
        epoch_loss /= len(pbar)
        return epoch_loss

    def evaluate(self, epoch):
        # evaluate by generation
        self.model.eval().requires_grad_(False)
        with open(os.path.join(self.log_dir, str(epoch) + ".txt"), "w") as f:
            for length in range(32, 1025, 32):
                f.write(f"Length: {length}" + "\n")
                xs = self.sampler.sample()
                for x in xs:
                    f.write(x + "\n")
                f.write("=" * 50 + "\n")
        # evaluate by log likelihood

    def loss_fn(self, batch, perturbed_batch=None, sampling_eps=1e-3):
        """
        Batch shape: [B, L] int. D given from graph
        """
        t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps

        sigma, dsigma = self.noise(t)

        if perturbed_batch is None:
            perturbed_batch = self.graph.sample_transition(batch, sigma[:, None])

        log_score = self.log_score_fn(perturbed_batch, sigma)
        loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        loss = dsigma[:, None] * loss

        return loss

    def log_score_fn(self, x: Tensor, t: Tensor):
        return self.model(x, t.reshape(-1))
