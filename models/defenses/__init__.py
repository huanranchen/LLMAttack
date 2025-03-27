"""
所有的防御需要继承BaseModel，并重写generate, forward, generate_by_ids这三个函数
"""
from .PrefixSuffixDefense import PromptATPrefix, ICD, SelfReminder
from .DetectionDefense import PerplexityDetectorDefense
from .SelfDiffTextPure import SelfDiffTextPureAbsorb, SelfDiffTextPureUniform
