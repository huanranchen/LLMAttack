from models import Vicuna15SafetyDetector, Llama3SafetyDetector, Llama2SafetyDetector, BaseModel
from models.SEDD import DiffTextPureUniform, DiffTextPureAbsorb


num_keywords = [  # 注意这个估计要往大了估计。可以往大，一定不能往小。 (或者估计average case也在bayesian bound那里比较合理）
    5,
    1,
    4,
    6,
    2,
    2,
    2,
    2,
    3,
    3,
    5,
    3,
    6,
    4,
    3,
    8,
    3,
    5,
    4,
    2,
    3,
    4,
    6,
    2,
    3,
    3,
    2,
    3,
    2,
    1,
    2,
    3,
    1,
    3,
    4,
    3,
    1,
    1,
    2,
    2,
    3,
    2,
    2,
    4,
    2,
    5,
    4,
    2,
    4,
    4,
]  # 一个好的估计（average case），考虑这个变成mask后意思会不会变。而不是变成反义词那种极端case
num_keywords = [i * 1 for i in num_keywords]
trivial_bound = [i - 1 for i in num_keywords]
trivial_bound.sort()
print(trivial_bound)
print("upper bound of average certified radii: ", sum(trivial_bound) / len(trivial_bound))

vicuna = DiffTextPureUniform(BaseModel(), purify_noise_level=0.1)
beta = 0.9
vicuna.purify_noise_level = beta
bayesian_bound = []
beta_bar = 1 - beta
for num_keyword in num_keywords:
    upper_pA = 1 - beta**num_keyword
    bayesian_bound.append(vicuna.certify_given_pA(upper_pA))

bayesian_bound.sort()
print(bayesian_bound)
print("Bayesian bound of average certified radii: ", sum(bayesian_bound) / len(bayesian_bound))

# Bayesian bound应该比Trivial bound更低。
# 0.1时为2.00，0.25时为2.10
