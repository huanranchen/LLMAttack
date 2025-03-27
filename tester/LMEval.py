import torch
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from typing import Tuple, Dict, Iterable
from models import BaseModel


__all__ = ["lm_eval_gsm8k", "lm_eval_mmlu", "lm_eval_truthfulqa"]


def lm_eval_gsm8k(
    model: BaseModel,
    task: str = "gsm8k_cot_llama",
    limit: int = 100,
    device="cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md
    """
    hf_model = HFLM(pretrained=model, tokenizer=model.tokenizer, device=device)
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=[task],
        limit=limit,
        batch_size=1,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        log_samples=True,
    )
    score = results["results"][task]["exact_match,flexible-extract"]
    print("GSM8k Score: ", score)
    print("-" * 10)
    return score


def lm_eval_mmlu(
    model: BaseModel,
    task: str = "mmlu",
    limit: int = 10,
    device="cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md
    """
    assert task in ["mmlu", "mmlu_continuation", "mmlu_generation"]
    hf_model = HFLM(pretrained=model, tokenizer=model.tokenizer, device=device)
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=[task],
        limit=limit,
        batch_size=1,
        # apply_chat_template=True,
    )
    results = results["results"]
    score = sum([result["acc,none"] for task_name, result in results.items()])
    score /= len(results)
    print("MMLU Score: ", score)
    print("-" * 10)
    return score


def lm_eval_truthfulqa(
    model: BaseModel,
    task: str = "truthfulqa_mc1",
    limit: int = 100,
    device="cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md
    """
    assert task in ["truthfulqa_mc1", "truthfulqa_mc2", "truthfulqa_gen"]
    hf_model = HFLM(pretrained=model, tokenizer=model.tokenizer, device=device)
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=[task],
        limit=limit,
        batch_size=1,
        # apply_chat_template=True,
    )
    score = results["results"][task]["acc,none"]
    print("TruthfulQA Score: ", score)
    print("-" * 10)
    return score
