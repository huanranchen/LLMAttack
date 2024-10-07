import shortuuid
import random
import torch
import time
import json
import os
from models import BaseModel
from fastchat.llm_judge.common import (
    load_questions,
    temperature_config,
    load_model_answers,
    load_judge_prompts,
    play_a_match_single,
    MatchSingle,
    Judge,
    NEED_REF_CATS,
)
from tqdm import tqdm


__all__ = ["test_mt_bench"]


def generate_mbtanswer(
    model: BaseModel,
    question_file="",
    answer_file="",
    question_begin=None,
    question_end=None,
    num_choices=1,
):
    if os.path.exists(answer_file):
        os.remove(answer_file)
    model_id = model.__class__.__name__
    conv = model.conv
    tokenizer = model.tokenizer

    # Generate questions prepared in question_file
    questions = load_questions(question_file, question_begin, question_end)
    random.shuffle(questions)

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            turns = []
            conv.messages.clear()
            for j in range(len(question["turns"])):
                qs = question["turns"][j]  # a single str
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = torch.tensor(tokenizer([prompt]).input_ids, device=model.device)

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                outputs = model.generate_by_input_ids(
                    input_ids,  # 1, L
                    max_length=model.generation_max_length * (j + 1) / len(question["turns"]),
                    do_sample=do_sample,
                    temperature=None,
                    top_p=None,
                    output_logits=True,
                    return_dict_in_generate=True,
                )

                output_ids = outputs.sequences.squeeze()
                logits = torch.cat(outputs.logits, dim=0)

                if conv.stop_token_ids:
                    stop_token_ids_index = [i for i, id in enumerate(output_ids) if id in conv.stop_token_ids]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                answer = tokenizer.decode(output_ids[input_ids.numel() :])

                if conv.stop_str and isinstance(conv.stop_str, list):
                    stop_str_indices = sorted(
                        [answer.find(stop_str) for stop_str in conv.stop_str if answer.find(stop_str) > 0]
                    )
                    if len(stop_str_indices) > 0:
                        answer = answer[: stop_str_indices[0]]
                elif conv.stop_str and answer.find(conv.stop_str) > 0:
                    answer = answer[: answer.find(conv.stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            answer = answer.replace(special_tok, "")
                    else:
                        answer = answer.replace(special_token, "")
                answer = answer.strip()

                if conv.name == "xgen" and answer.startswith("Assistant:"):
                    answer = answer.replace("Assistant:", "", 1).strip()

                conv.update_last_message(answer)
                turns.append(answer)
                choices.append({"index": i, "turns": turns})

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)

        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


# This one has something wrong, since we didn't use dir name to pass the params
def check_data(questions, model_answers, ref_answers, models, judges):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert q["question_id"] in m_answer, f"Missing model {m}'s answer to Question {q['question_id']}"
    # check ref answers
    for jg in judges.values():
        if not jg.ref_based:
            continue
        for q in questions:
            if q["category"] not in NEED_REF_CATS:
                continue
            assert (
                q["question_id"] in ref_answers[jg.model_name]
            ), f"Missing reference answer to Question {q['question_id']} for judge {jg.model_name}"


def make_match_single(
    questions,
    model_id,
    model_answers,
    judge,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        q_id = q["question_id"]
        m = model_id
        a = model_answers[m][q_id]
        if ref_answers is not None:
            ref = ref_answers[judge.model_name][q_id]
            matches.append(MatchSingle(dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn))
        else:
            matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True)
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def generate_mbtscores(
    model: BaseModel,
    question_file="",
    ref_answer_dir="",
    answer_file="",
    judge_file="",
    question_begin=None,
    question_end=None,
    num_choices=1,
):
    model_id = model.__class__.__name__
    judge_model = "gpt-4"
    output_file = f"resources/mtbench/model_judgment/gpt4_single.jsonl"

    # Prepare all txt
    questions = load_questions(question_file, question_begin, question_end)
    model_answers = load_model_answers(answer_file)

    ref_answers = load_model_answers(ref_answer_dir)
    judge_prompts = load_judge_prompts(judge_file)

    # Prepare Judge Prompt
    judges = make_judge_single(judge_model, judge_prompts)

    # # Check data
    check_data(questions, model_answers, ref_answers, [model_id], judges)

    # Distinguish math and default questions
    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    if os.path.exists(output_file):
        os.remove(output_file)

    make_match_func = make_match_single
    play_a_match_func = play_a_match_single

    # Make mathes
    matches = []
    matches += make_match_func(question_default, model_id, model_answers, judges["default"])
    # matches += make_match_func(
    #     question_math,
    #     model_id,
    #     model_answers,
    #     judges["math"],
    #     ref_answers,
    # )
    # matches += make_match_func(
    #     question_default,
    #     model_id,
    #     model_answers,
    #     judges["default-mt"],
    #     multi_turn=True,
    # )
    # matches += make_match_func(
    #     question_math,
    #     model_id,
    #     model_answers,
    #     judges["math-mt"],
    #     ref_answers,
    #     multi_turn=True,
    # )
    match_stat = {
        "bench_name": "mt",
        "mode": "single",
        "judge": "gpt-4",
        "baseline": None,
        "model_list": model_id,
        "total_num_questions": len(questions),
        "total_num_matches": len(matches),
        "output_path": output_file,
    }

    # Output to a file with the judgement description
    for match in tqdm(matches):
        play_a_match_func(match, output_file=output_file)

    # Get the average scores
    total_score = 0
    count = 0

    with open(output_file, "r") as f:
        for line in f.readlines():
            dic = json.loads(line)
            assert dic["score"] != -1, "gpt judgement error"
            total_score += dic["score"]
            count += 1

    average_score = total_score / count if count > 0 else 0

    return average_score


def test_mt_bench(model: BaseModel, question_begin: int = None, question_end: int = None) -> float:
    question_file = "resources/mtbench/question.jsonl"
    answer_file = "resources/mtbench/tmp_outputs/" + model.__class__.__name__ + ".jsonl"
    ref_answer_dir = "resources/mtbench/ref_answers/"
    judge_file = "resources/mtbench/judge_prompts.jsonl"

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    generate_mbtanswer(
        model,
        question_file=question_file,
        answer_file=answer_file,
        question_begin=question_begin,
        question_end=question_end,
        num_choices=1,
    )

    res = generate_mbtscores(
        model,
        question_file=question_file,
        ref_answer_dir=ref_answer_dir,
        answer_file=os.path.dirname(answer_file),
        judge_file=judge_file,
        question_begin=question_begin,
        question_end=question_end,
        num_choices=1,
    )
    print(f"model {model.__class__}, MTBench scores: {res}")
    return res
