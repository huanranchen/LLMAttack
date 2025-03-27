
# Official Code Implementation of "Toward the Worst-case Robustness of LLMs"

---

## What We Have Done in the Paper

### Implemented I-I-GCG Attacks, Demonstrating Near 0% Robustness for Existing Defenses

We do not propose advanced optimizers in this paper. Our 100% attack success rate—unlike previous works—is achieved by strictly ensuring token consistency during both optimization and inference. Previous approaches failed to enforce this consistency, resulting in adversarial tokens with low loss during training but higher loss during inference due to subtle tokenization differences. These methods effectively operated in grey-box settings rather than true white-box settings. Token consistency is the key factor enabling our perfect success rate. Other techniques in this paper (e.g., attacking longer sequences, removing gradients, warm starts) are incremental improvements designed to accelerate attacks or handle edge cases, such as transitions to safe responses. While token consistency is simple in principle, ensuring it across every model, defense, and even individual sentence required significant effort.

| **I²-GCG**       | **No Defense** | **PPL** | **ICD** | **Self Reminder** | **PAT** | 
|-------------------|----------------|---------|---------|-------------------|---------|
| **Vicuna-7B**    | 0%             | 0%      | 0%      | 0%                | 0%      |
| **Llama2-7B**    | 0%             | 0%      | 0%      | 0%                | 2%      | 
| **Llama3-8B**    | 0%             | 0%      | 0%      | 0%                | 0%      | 

Similar to adversarial robustness in computer vision, only a few defenses—like adversarial training and randomized smoothing—avoid 0% worst-case robustness. In vision, most defenses have been proven ineffective, ultimately attacked to 0% robustness, except for adversarial training and randomized smoothing (including purification-based defenses). We reach a nearly identical conclusion here. While adversarial training may partially mitigate this issue, current approaches focus more on alignment than traditional, extensive adversarial training. As a result, they fail to address worst-case robustness, offering only minor improvements in average-case robustness.

### White-box Evaluations Provide an Upper Bound for Worst-case Robustness, While Certified Robustness Serves as the Lower Bound

White-box evaluations provide an upper bound for worst-case robustness, which may decrease with future, stronger attacks. In contrast, certified robustness offers a theoretical lower bound, which may increase with advancements in certification analysis. We believe that as researchers refine evaluation and certification methods, the gap between the empirical upper bound and theoretical lower bound will gradually narrow.

### Certified Robustness as a Fractional Knapsack or 0-1 Knapsack Problem

When the base function \(f\) is bounded, randomized smoothing becomes a fractional knapsack problem. If \(f\) is binary, it transforms into a 0-1 knapsack problem, potentially improving the certified bound.

---

## What You Can Do with This Package

### Attack Existing Defenses Using Our Implemented Optimizers

We have implemented various attackers, including GCG, I-GCG, I-I-GCG, ICA, AutoDAN, and more. These codes can help you understand and run these attackers more easily.

Reference codes are located in:
```shell
./experiments/attacks
```

### Refer to Our Fractional Knapsack or 0-1 Knapsack Implementation

**Note:** Our code cannot directly provide a certified lower bound for your randomized defenses, as this is impossible without tailoring to specific kernels due to varying trading rates. However, you can reference our implementation, which we believe is straightforward to adapt.

```shell
./models/defenses/SelfDiffTextPure.py
./models/SEDD/DiffTextPure.py
```

```python
def fractional_knapsack(self, pA: float, beta: float, d: int, V: int = 30000) -> float:
    """
    :param pA: lower confidence bound
    :param beta: the word that would change to **a specific word** at time t
    :param d: adv suffix length
    :param V: vocabulary size
    :return: certified robustness bound
    """
    # Step 1. Compute ratio and volume
    volume, ratio = self.compute_volume_and_ratio(beta, d, V)
    measure_on_adv = [v * r for v, r in zip(volume, ratio)]
    # Step 2. Solve Fractional Knapsack
    all_p_ori = 0
    p_adv = 0
    for i, cur_volume in enumerate(volume):
        if all_p_ori + cur_volume < pA:
            all_p_ori += cur_volume
            p_adv += cur_volume * ratio[i]
        else:  # all_p_ori + cur_volume >= pA
            delta = pA - all_p_ori
            p_adv += delta * ratio[i]
            return p_adv
    return p_adv
```

### Use Our Implemented (Accelerated) Absorbing/Uniform/Gaussian Kernels

We have implemented accelerated versions of these kernels (see Sections 5.2 and 5.3 in our paper) by pre-computing trading rates, categorizing items based on trading rates, and calculating the number of items in each category.

```shell
./models/defenses/SelfDiffTextPure.py
./models/SEDD/DiffTextPure.py
```

---

## Reference

If you find this repository helpful, please cite our paper:

```
@article{chen2025towards,
  title={Towards the Worst-case Robustness of Large Language Models},
  author={Chen, Huanran and Dong, Yinpeng and Wei, Zeming and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2501.19040},
  year={2025}
}
```

We strongly recommend interested readers start with our blog, which is more accessible than the paper and allows direct discussion via comments:

```
证明LLM的鲁棒性下界就是在解0-1背包问题？ - 虚无的文章 - 知乎
https://zhuanlan.zhihu.com/p/21266930786
```

---

This version ensures proper Markdown syntax, eliminates grammatical errors, and enhances readability while keeping the technical content intact. Let me know if you need further adjustments!