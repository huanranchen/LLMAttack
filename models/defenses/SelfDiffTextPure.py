import torch
import random
from models import BaseModel
from typing import Union, List, Tuple
from scipy.special import comb
from statsmodels.stats.proportion import proportion_confint


def binomial(n: int, m: int) -> int:
    assert n >= m
    return comb(n, m, exact=True)


class SelfDiffTextPure(BaseModel):
    def __init__(self, purifier: BaseModel, generator: BaseModel, beta: float = 0.25, verbose: bool = False):
        super(SelfDiffTextPure, self).__init__(generator.model, generator.tokenizer, generator.conv, generator.device)
        self.purifier = purifier
        self.generator = generator
        self.beta = beta
        self.verbose = verbose
        if not hasattr(self.purifier, "tokenizer") or self.purifier.tokenizer is None:
            self.purifier.tokenizer = self.generator.tokenizer

    def forward_noising(self, question: str) -> str:
        pass

    def generate(
        self,
        question: Union[str, List[str]],
        *args,
        prefix: Union[str, List[str]] = "",
        suffix: Union[str, List[str]] = "",
        return_purified_result: bool = False,
        **kwargs,
    ) -> Union[Tuple[str], Tuple[List[str], List[str]]]:
        if isinstance(question, str):
            question = [question]
        if isinstance(prefix, str):
            prefix = [prefix] * len(question)
        if isinstance(suffix, str):
            suffix = [suffix] * len(question)
        question = [self.forward_noising(q) for q in question]
        question = [p + q + s for p, q, s in zip(prefix, question, suffix)]
        purified_question = [self.purifier.generate(q) for q in question]  # condition on prefix and suffix
        if self.verbose:
            print("original question: ", question)
            print("purified question: ", purified_question)
        inputs = purified_question
        answer = [self.generator.generate(q, *args, **kwargs) for q in inputs]
        # answer = answer[0] if len(answer) == 1 else answer
        return (answer, purified_question) if return_purified_result else answer

    @staticmethod
    def get_pA_given_n_and_nA(n: int = 10000, nA: int = 10000, alpha: float = 0.01):
        return proportion_confint(nA, n, alpha=2 * alpha, method="beta")[0]  # single side Clopper-Pearson


class SelfDiffTextPureAbsorb(SelfDiffTextPure):
    def __init__(self, *args, **kwargs):
        super(SelfDiffTextPureAbsorb, self).__init__(*args, **kwargs)
        self.mask_tokens = self.purifier.tokenizer("[MASK]", add_special_tokens=False).input_ids

    def forward_noising(self, question: str) -> str:
        tokenized = self.purifier.tokenizer(question, add_special_tokens=False).input_ids
        result = []
        for i in tokenized:
            if random.random() < self.beta:
                result += self.mask_tokens
            else:
                result.append(i)
        return self.purifier.tokenizer.decode(result)

    def compute_difftextpure_absorb_min_adv_output(self, pA: float, beta: float, d: int):
        beta_to_the_d = beta**d
        if pA < 1 - beta_to_the_d:
            return 0
        return pA - (1 - beta_to_the_d)

    def certify_given_pA(self, pA: float, threshold: float = 4.6e-5 * args, **kwargs) -> int:
        assert 0 <= pA <= 1
        beta, beta_bar = self.beta, 1 - self.beta
        for l0_diff in range(1, 1024):
            p_adv = self.compute_difftextpure_absorb_min_adv_output(pA, beta, l0_diff)
            # print(l0_diff, p_adv)
            if p_adv < threshold:
                return l0_diff - 1
        return 1023


class SelfDiffTextPureUniform(SelfDiffTextPure):
    def __init__(self, *args, **kwargs):
        super(SelfDiffTextPureUniform, self).__init__(*args, **kwargs)

    def forward_noising(self, question: str) -> str:
        tokenized = self.purifier.tokenizer(question, add_special_tokens=False).input_ids
        for i in range(len(tokenized)):
            if random.random() < self.beta:
                tokenized[i] = random.randint(0, self.purifier.tokenizer.vocab_size)
        return self.purifier.tokenizer.decode(tokenized)

    def certify_given_pA(self, pA: float, dim: int = None, threshold: float = 4.6e-5) -> int:
        dim = dim or self.purifier.tokenizer.vocab_size
        assert 0 <= pA <= 1
        alpha, beta_bar = self.beta / dim, 1 - self.beta
        for l0_diff in range(1, 1024):
            p_adv = self.compute_difftextpure_uniform_min_adv_output(pA, alpha, l0_diff, dim)
            # print(l0_diff, p_adv)
            if p_adv < threshold:
                return l0_diff - 1
        return 1023

    def compute_difftextpure_uniform_min_adv_output(self, pA: float, alpha: float, d: int, V: int = 30000) -> float:
        """

        :param pA: lower confidence bound
        :param alpha: the word that would change to **a specific word** at time t
        :param d: adv suffix length
        :param V: vocabulary size
        :return:
        """
        # Step 1. Compute ratio and volume
        volume, ratio = self.compute_volume_and_ratio(alpha, d, V)
        measure_on_adv = [v * r for v, r in zip(volume, ratio)]
        # Step 2. Solving Fractal Knapsack
        all_p_ori = 0
        p_adv = 0
        # print(volume)
        # print(measure_on_adv)
        for i, cur_volume in enumerate(volume):
            if all_p_ori + cur_volume < pA:
                all_p_ori += cur_volume
                p_adv += cur_volume * ratio[i]
            else:  # all_p_ori + p_ori >= pA
                delta = pA - all_p_ori
                p_adv += delta * ratio[i]
                return p_adv
        return p_adv

    @staticmethod
    def compute_volume_and_ratio(alpha: float, d: int, V: int = 30000):
        beta_bar = 1 - (V - 1) * alpha  # 不变的概率
        ratio, volume = [], []
        for i in range(d + 1):  # normal suffix. 0-d
            for j in range(d + 1):  # adv suffix. 0-d
                if i + j >= d:  # 对抗样本扰动的加上正常样本扰动的，应该能使得d个位置都扰动
                    can_change = i + j - d
                    cant_change = i - can_change
                    # 不可以改变的里，是C(i, cant_change)种位置选择。剩下的可以改变的里，每个有(V - 2)个可能。
                    num = binomial(d, i) * binomial(i, cant_change) * 1 * (V - 2) ** can_change
                    cur_ori_prob = alpha**i * beta_bar ** (d - i)
                    cur_adv_prob = alpha**j * beta_bar ** (d - j)
                    ratio.append(cur_adv_prob / cur_ori_prob)
                    volume.append(cur_ori_prob * num)
                    # print(i, j)
        sorted_pairs = sorted(zip(volume, ratio), key=lambda pair: pair[1])
        volume = [i[0] for i in sorted_pairs]
        ratio = [i[1] for i in sorted_pairs]
        return volume, ratio
