import torch
import os
from torch import nn, Tensor
from torch.utils.data import DataLoader
from ..samplers.NoiseSchedule import Noise
from ..samplers.Graphs import Graph
from ..samplers import BaseSampler
from transformers import PreTrainedTokenizer
from tqdm import tqdm
from huggingface_hub import PyTorchModelHubMixin


class SEDDTrainer:
    def __init__(
        self,
        model: PyTorchModelHubMixin,
        tokenizer: PreTrainedTokenizer,
        noise: Noise,
        graph: Graph,
        sampler: BaseSampler,
        verbose: bool = False,
        device=torch.device("cuda"),
        log_dir: str = "./log_dir/",
        eval_mode: str = "epoch_eval",
    ):
        assert eval_mode in ["epoch_eval", "step_eval"]
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
        self.eval_mode = eval_mode

    def train(self, loader: DataLoader, total_epoch: int = 100, ddp=False, *args, **kwargs) -> None:
        self.model.train().requires_grad_(True)
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        model = torch.nn.parallel.DistributedDataParallel(self.model, [self.device]) if ddp else self.model
        for epoch in range(1, total_epoch + 1):
            loss = self.train_an_epoch(model, loader, optimizer, scaler, epoch, ddp=ddp, *args, **kwargs)
            print(f"epoch: {epoch}, loss: {loss:.4f}")
            if self.eval_mode == "epoch_eval":
                self.evaluate(epoch + 1, 0, ddp)

    def train_an_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        epoch: int,
        eval_frequency: int = 10000,
        ddp: bool = False,
        gradient_accumulation: int = 1,
    ) -> float:
        model.train().requires_grad_(True)
        pbar = tqdm(loader)
        epoch_loss = 0
        for step, (x, masks) in enumerate(pbar, 1):
            x, masks = x.to(self.device), masks.to(self.device)
            loss = (self.loss_fn(model, x) * masks).sum(-1).mean() / gradient_accumulation
            if step % gradient_accumulation == 0:
                optimizer.zero_grad()
            scaler.scale(loss).backward()
            if step % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            epoch_loss += loss.item()
            if step % 10 == 0:
                pbar.set_postfix_str(f"step {step}, loss {epoch_loss / step}")
            if self.eval_mode == "step_eval" and step % eval_frequency == 0:
                self.evaluate(epoch, step, ddp)
                model.train().requires_grad_(True)
        epoch_loss /= len(pbar)
        return epoch_loss

    def evaluate(self, epoch: int, step: int, ddp: bool):
        self.model.eval().requires_grad_(False)
        if not ddp or torch.distributed.get_rank() == 0:
            # 1. save state dict
            self.model.save_pretrained(os.path.join(self.log_dir, f"epochs_{epoch}_steps_{step}_ckpt"))
            # torch.save(self.model.state_dict(), os.path.join(self.log_dir, f"epochs_{epoch}_steps_{step}.pth"))
            print(f"Checkpoint at epochs_{epoch}_steps_{step} saved")
            # 2. evaluate by generation
            with open(os.path.join(self.log_dir, f"epochs_{epoch}_steps_{step}.txt"), "w") as f:
                for length in tqdm([32, 64, 128, 256, 512, 1024]):
                    f.write(f"Length: {length}" + "\n" * 2)
                    xs = self.sampler.sample(batch_size=4, length=length)
                    for x in xs:
                        f.write(x + "\n" + "-" * 50 + "\n")
                    f.write("=" * 200 + "\n" * 5)
            print(f"Evaluation finished, save at epochs_{epoch}_steps_{step}.txt")
            # 3. TODO: evaluate by log likelihood
        # 4. barrier
        if ddp:
            torch.distributed.barrier()

    def loss_fn(self, model: nn.Module, batch, perturbed_batch=None, sampling_eps=1e-3):
        """
        Batch shape: [B, L] int. D given from graph
        """
        t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps

        sigma, dsigma = self.noise(t)

        if perturbed_batch is None:
            perturbed_batch = self.graph.sample_transition(batch, sigma[:, None])

        log_score = self.log_score_fn(model, perturbed_batch, sigma)
        loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        loss = dsigma[:, None] * loss

        return loss

    @staticmethod
    def log_score_fn(model: nn.Module, x: Tensor, t: Tensor):
        return model(x, t.reshape(-1))
