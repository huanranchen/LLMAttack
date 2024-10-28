import torch
from typing import Tuple, Callable
from torch import Tensor
from transformers import GPT2TokenizerFast
from scipy.special import comb
from statsmodels.stats.proportion import proportion_confint
from ..BaseModel import BaseModel
from .samplers import BaseSampler, EulerSEDDSampler, LogLinearNoise, Uniform, Absorbing
from .SEDDBackbone import SEDD


__all__ = ["DiffTextPureUniform", "DiffTextPureAbsorb", "DiffTextPure"]


def binomial(n: int, m: int) -> int:
    assert n >= m
    return comb(n, m, exact=True)


class DiffTextPure(BaseModel):
    def __init__(
        self, model: BaseModel, sampler: BaseSampler, *args, verbose: bool = False, purify_batch_size=4, **kwargs
    ):
        super(DiffTextPure, self).__init__(model.model, model.tokenizer, model.conv, *args, **kwargs)
        self.to_protect_model = model
        self.sampler = sampler
        self.verbose = verbose
        # use to_protect_model's function
        self.get_prompt = self.to_protect_model.get_prompt
        self.purify_batch_size = purify_batch_size

    def generate(self, question: str, *args, **kwargs) -> str or Tuple[str, Tensor]:
        purified_question = self.sampler([question])[0]
        if self.verbose:
            print("original question: ", question)
            print("purified question: ", purified_question)
        return self.to_protect_model.generate(purified_question, *args, **kwargs)

    def generate_by_input_ids(self, input_ids: Tensor, *args, **kwargs) -> Tensor:
        """
        given input ids, return generate results. Similar to huggingface generate.
        :param input_ids: 1, L
        :return: output_ids
        """
        input_ids = input_ids.squeeze()
        assert input_ids.ndim == 1, "Currently DiffTextPure only supports one instance."
        # Step 1. get the question parts.
        decoded_text = self.tokenizer.decode(input_ids)
        parts = decoded_text.split("\n### Human:")
        last_question = parts[-1].split("\n### Assistant:")[0].strip()
        # Step 2. DiffPure
        purified_question = self.sampler([last_question])[0]
        # Step 3. Concatenate back
        purified_text = "\n### Human:".join(parts[:-1]) + "\n### Human:" + purified_question + "\n### Assistant:"
        purified_ids = torch.tensor(
            self.tokenizer.encode(purified_text, add_special_tokens=False), device=input_ids.device
        ).unsqueeze(0)
        # Step 2. Generate by ids.
        return self.to_protect_model.generate_by_input_ids(purified_ids, *args, **kwargs)

    def forward(self, input_ids: Tensor, *args, **kwargs):
        """
        这样做似乎是会有问题的，会不会purify之后多了个token，但是truncate的token错位了
        :param input_ids: B, L
        :return:
        """
        # Step 1: Split the question part
        role0, role1 = self.conv.roles[0], self.conv.roles[1]
        texts = self.tokenizer.batch_decode(input_ids)  # List[str]
        questions = []
        for text in texts:  # + 1 to remove the space. e.g., Human: Hi. There are space before "Hi".
            start = text.find(role0) + len(role0) + 1
            end = text.find(role1, start)  # search from start
            questions.append(text[start:end])
        # ------------------------------------------------------------------------------------------
        # Step 2: purify the question part
        purified_questions = [
            q
            for i in range(0, len(questions), self.purify_batch_size)
            for q in self.sampler(questions[i : i + self.purify_batch_size])
        ]
        purified_ids = self.tokenizer.batch_encode_plus(purified_questions, add_special_tokens=False).input_ids  # List
        # ------------------------------------------------------------------------------------------
        # Step 3: Truncate question part into origin length. (maximum end_indices[i]-begin_indices[i])
        # If less than this, then pad by padding token
        # purified_ids is List[List[int]]
        begin_indices = [
            len(self.tokenizer(text[: text.find(role0) + len(role0)], add_special_tokens=False).input_ids)
            for text in texts
        ]  # List[int]
        end_indices = [
            len(self.tokenizer(text[: text.find(role1)], add_special_tokens=False).input_ids) for text in texts
        ]
        # Truncating
        purified_ids = [ids[: end_indices[i] - begin_indices[i]] for i, ids in enumerate(purified_ids)]
        # Padding
        purified_ids = [
            ids + (end_indices[i] - begin_indices[i] - len(ids)) * [self.tokenizer.pad_token_id]
            for i, ids in enumerate(purified_ids)
        ]
        # ------------------------------------------------------------------------------------------
        # Step 4: Concatenate back
        ids = [
            input_id[: begin_indices[i]] + purified_ids[i] + input_id[end_indices[i] :]
            for i, input_id in enumerate(input_ids.tolist())
        ]  # List of list
        ids = [each_ids[: input_ids.shape[1]] for each_ids in ids]
        # ------------------------------------------------------------------------------------------
        # Step 5: use the purified one to get the result
        ids = torch.tensor(ids, device=self.device)
        return self.to_protect_model.forward(ids, *args, **kwargs)


class DiffTextPureAbsorb(DiffTextPure):
    def __init__(self, model: BaseModel, *args, **kwargs):
        transformer = SEDD.from_pretrained("louaaron/sedd-small")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Absorbing(50257))
        super(DiffTextPureAbsorb, self).__init__(model, sampler, *args, **kwargs)


class DiffTextPureUniform(DiffTextPure):
    def __init__(self, model: BaseModel, *args, **kwargs):
        transformer = SEDD.from_pretrained("./resources/checkpoints/SEDD/uniform_small/huanran_repaired/")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        sampler = EulerSEDDSampler(transformer, tokenizer, LogLinearNoise(), Uniform(50257))
        super(DiffTextPureUniform, self).__init__(model, sampler, *args, **kwargs)

    @staticmethod
    def compute_difftextpure_uniform_min_adv_output_weak(pA: float, beta: float, d: int) -> float:
        """

        :param pA: lower confidence bound
        :param beta: the word that would change to **a specific word** at time t
        :param d: adv suffix length
        :return:
        """
        assert 0 <= beta <= 1, 0 <= pA <= 1
        beta = beta if beta < 0.5 else (1 - beta)
        p_oris = [comb(d, i, exact=True) * beta**i * (1 - beta) ** (d - i) for i in range(d + 1)]
        ratio = [beta**d / beta**i * (1 - beta) ** (d - i) for i in range(d + 1)]
        sorted_pairs = sorted(zip(p_oris, ratio), key=lambda pair: pair[1], reverse=False)  # 似乎有问题，应该按照ratio排序，这里写错了
        p_oris = [i[0] for i in sorted_pairs]
        ratio = [i[1] for i in sorted_pairs]

        all_p_ori = 0
        p_adv = 0
        for i, p_ori in enumerate(p_oris):
            if all_p_ori + p_ori < pA:
                all_p_ori += p_ori
                p_adv += p_ori * ratio[i]
            else:  # all_p_ori + p_ori >= pA
                delta = pA - all_p_ori
                p_adv += delta * ratio[i]
                return p_adv
        return p_adv

    @staticmethod
    def compute_difftextpure_uniform_min_adv_output(pA: float, beta: float, d: int, V: int = 30000) -> float:
        """

        :param pA: lower confidence bound
        :param beta: the word that would change to **a specific word** at time t
        :param d: adv suffix length
        :param V: vocabulary size
        :return:
        """
        beta_bar = 1 - (V - 1) * beta  # 不变的概率
        ratio, volume = [], []
        for i in range(d + 1):  # normal suffix. 0-d
            for j in range(d + 1):  # adv suffix. 0-d
                if i + j >= d:  # 对抗样本扰动的加上正常样本扰动的，应该能使得d个位置都扰动
                    can_change = i + j - d
                    cant_change = i - can_change
                    # 不可以改变的里，是C(i, cant_change)种位置选择。剩下的可以改变的里，每个有(V - 2)个可能。
                    num = binomial(d, i) * binomial(i, cant_change) * 1 * (V - 2) ** can_change
                    cur_ori_prob = beta**i * beta_bar ** (d - i)
                    cur_adv_prob = beta**j * beta_bar ** (d - j)
                    ratio.append(cur_adv_prob / cur_ori_prob)
                    volume.append(cur_ori_prob * num)
        sorted_pairs = sorted(zip(volume, ratio), key=lambda pair: pair[1])
        volume = [i[0] for i in sorted_pairs]
        ratio = [i[1] for i in sorted_pairs]
        all_p_ori = 0
        p_adv = 0
        for i, cur_volume in enumerate(volume):
            if all_p_ori + cur_volume < pA:
                all_p_ori += cur_volume
                p_adv += cur_volume * ratio[i]
            else:  # all_p_ori + p_ori >= pA
                delta = pA - all_p_ori
                p_adv += delta * ratio[i]
                return p_adv
        return p_adv

    def compute_beta_given_t(self, noise_level_t: float) -> float:
        """

        :param noise_level_t: diffusion time.
        :return:
        """
        sigma_bar_t, dsigma = self.sampler.noise.forward(torch.tensor(noise_level_t))
        V = self.sampler.graph.dim
        exp_sigma_bar_t = torch.exp(sigma_bar_t)
        beta = (exp_sigma_bar_t - 1) / (V * exp_sigma_bar_t)
        return beta.item()

    def get_pA(self, text: str, judge: Callable, n: int = 10000, alpha: float = 0.01):
        result = 0
        for _ in range(n):
            answer = self.generate(text)
            result += judge(answer)
        upper_bound = result / n  # Just for reference
        pA = proportion_confint(result, n, alpha=2 * alpha, method="beta")[0]  # single side Clopper-Pearson
        return pA

    def certify_given_pA(self, pA: float) -> int:
        assert 0 <= pA <= 1
        beta = self.compute_beta_given_t(noise_level_t=0.25)  # Attention! 这里需要和实际用的保持一致  # 问题：sigma_t是否选的使得t恰好就是概率？
        for adv_suffix_length in range(1, 1024):
            r = self.compute_difftextpure_uniform_min_adv_output(pA, beta, adv_suffix_length, self.sampler.graph.dim)
            print(adv_suffix_length, r)
            if r < 0.5:
                return adv_suffix_length - 1
        return 1023

    def certify(self, text: str, judge: Callable, n: int = 10000) -> int:
        """

        :param text:
        :param judge:e.g.,正确返回1，错误返回0。安全返回1，不安全返回0
        :param n:
        :return:
        """
        pA = self.get_pA(text, judge, n)
        return self.certify_given_pA(pA)
