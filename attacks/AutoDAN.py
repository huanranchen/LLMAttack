import torch
from .GCG import get_embedding_matrix, get_embeddings, get_nonascii_toks, get_filtered_cands, GCGAttack
from torch import nn
import numpy as np


class AutoDAN(GCGAttack):
    """
    AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models
    """

    def __init__(
        self,
        *args,
        w1: float = 3,
        w2: float = 100,
        tau: float = 1,
        num_steps: int = 100,
        batch_size: int = 512,  # 理论上加了kv cache后这个可以巨大
        batch_size_for_calculating_loss: int = 16,
        **kwargs
    ):
        super(AutoDAN, self).__init__(
            *args,
            num_steps=num_steps,
            batch_size=batch_size,
            batch_size_for_calculating_loss=batch_size_for_calculating_loss,
            **kwargs
        )
        self.w1, self.w2, self.tau = w1, w2, tau
        self.past_key_values = ()

    def attack(self):
        model = self.models[0]
        not_allowed_tokens = get_nonascii_toks(model.tokenizer)
        adv_suffix = ""
        for step in range(1, self.num_steps + 1):
            # Step 0. Add a new token.
            adv_suffix = adv_suffix + "!"  # 这个!也可以继续修改成别的token，可能效果更好。
            # 但是如果修改成[，一旦出现[[，由于llama2中[[和[token个数相同，就会死循环
            # Step 1. Tokenization.
            input_ids, grad_slice, target_slice, loss_slice = model.get_prompt(self.prompt, adv_suffix, self.target)
            # Step 2. Compute Coordinate Gradient
            coordinate_grad, topk, loss = self.token_gradients(input_ids, grad_slice, target_slice, loss_slice)
            # Step 5. Verbose
            if self.verbose:
                self.verbose_info(step, loss, adv_suffix, input_ids, grad_slice, target_slice, loss_slice, topk)
            # grad_slice = slice(grad_slice.start + 1, grad_slice.stop + 1)
            # Step 3. Sample a batch of new tokens based on the coordinate gradient.
            # Notice that we only need the one that minimizes the loss.
            adv_suffix, _ = self.enumerate_best_token(
                input_ids, adv_suffix, target_slice, grad_slice, coordinate_grad, not_allowed_tokens
            )
            # Step 4. Check Success
            if (loss < 0.5 or step % 10 == 0) and self.check_success(adv_suffix, input_ids[target_slice]):
                return self.prompt + " " + adv_suffix
        return self.prompt + " " + adv_suffix

    def token_gradients(self, input_ids, input_slice, target_slice, loss_slice):
        """
        Computes gradients of the loss with respect to the coordinates.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input sequence in the form of token ids.
        input_slice : slice
            The slice of the input sequence for which gradients need to be computed.
        target_slice : slice
            The slice of the input sequence to be used as targets.
        loss_slice : slice
            The slice of the logits to be used for computing the loss.

        Returns
        -------
        torch.Tensor
            The gradients of each token in the input_slice with respect to the loss.
        """
        model = self.models[0].model
        embed_weights = get_embedding_matrix(model)
        one_hot = torch.zeros(
            input_ids[input_slice].shape[0], embed_weights.shape[0], device=model.device, dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1,
            input_ids[input_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype),
        )
        one_hot.requires_grad_()
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)

        # now stitch it together with the rest of the embeddings
        embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
        full_embeds = torch.cat(
            [embeds[:, : input_slice.start, :], input_embeds, embeds[:, input_slice.stop :, :]], dim=1
        )

        out = model(inputs_embeds=full_embeds, use_cache=True)
        logits = out.logits
        # Do not save last token.
        self.past_key_values = tuple(
            (key_value[0][:, :, : input_slice.stop - 1, :], key_value[1][:, :, : input_slice.stop - 1, :])
            for key_value in out.past_key_values
        )
        probs = torch.softmax(logits, dim=-1).squeeze()
        targets = input_ids[target_slice]
        loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
        loss.backward()

        grad1 = one_hot.grad.clone()[-1].unsqueeze(0)  # 1, |V|
        grad2 = probs[input_slice.stop - 2].unsqueeze(0)  # 1, |V|  倒数第二个token的输出是倒数第一个词的概率
        grad = -self.w1 * grad1 + grad2  # 1, |V|

        # Compute topk
        prediction = torch.topk(logits[0, loss_slice, :], k=10, dim=1)[1]  # L, K
        position_table = prediction == targets.unsqueeze(1)
        topk = torch.max(position_table, dim=1)[1]
        topk = torch.where(position_table.sum(1) != 0, topk, float("inf"))
        return grad, topk, loss

    @staticmethod
    def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):
        """
        :param control_toks: Length of adv suffix now. L.
        :param grad: 1, |V|
        :param batch_size:
        :param topk:
        :param temp:
        :param not_allowed_tokens:
        :return:
        """
        topk = max(topk, batch_size)  # 这里只需要enumerate一个token，因此topk应该取二者最大值
        if not_allowed_tokens is not None:
            grad[:, not_allowed_tokens.to(grad.device)] = -np.infty

        # 这部分是越大越好。因为我前面就给了梯度负号和概率正号。概率越大越好，梯度越负（越小）越好。
        top_indices = grad.topk(topk, dim=1).indices  # 1, B
        new_control_toks = torch.stack([control_toks] * batch_size)  # B, L
        # 最后一个token替换为top indices中的一个
        new_control_toks[:, -1] = top_indices.squeeze()  # B, L
        return new_control_toks  # B, L

    @torch.no_grad()
    def batch_calculate_loss(self, *, model, input_ids, attention_mask, target_slice, control_slice, batch_size=512):
        losses = []
        for i in range(0, input_ids.shape[0], batch_size):
            batch_input_ids = input_ids[i : i + batch_size]
            if attention_mask is not None:
                batch_attention_mask = attention_mask[i : i + batch_size]
            else:
                batch_attention_mask = None
            B = batch_input_ids.shape[0]
            past_key_values = tuple(
                (key_value[0].expand(B, -1, -1, -1), key_value[1].expand(B, -1, -1, -1))
                for key_value in self.past_key_values
            )
            outputs = model(
                input_ids=batch_input_ids[:, control_slice.stop - 1 :],
                attention_mask=batch_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logit = outputs.logits
            losses.append(self.target_loss(batch_input_ids, logit, target_slice, control_slice))

        return torch.cat(losses, dim=0)

    def target_loss(self, batch_input_ids, logits, target_slice: slice, control_slice: slice):
        """
        w2 * loss1 + 1 * loss2
        batch_input_ids is B, L
        """
        # 因为使用了kv cache，为了不改变原来的代码，直接padding回去，这样最简单
        padding = torch.zeros((batch_input_ids.shape[0], control_slice.stop - 1, logits.shape[2]), device=logits.device)
        logits = torch.cat([padding, logits], dim=1)  # B, L, |V|
        # 后面是原来的代码
        crit = nn.CrossEntropyLoss(reduction="none")
        loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
        loss1 = crit(logits[:, loss_slice, :].transpose(1, 2), batch_input_ids[:, target_slice]).mean(dim=-1)

        probs = torch.softmax(logits[:, control_slice.stop - 2, :], dim=-1)  # B, |V|
        batch_adv_ids = batch_input_ids[:, control_slice.stop - 1].unsqueeze(1)  # B, 1
        loss2 = probs.gather(1, batch_adv_ids).squeeze()  # B
        return loss1 * self.w2 - loss2
