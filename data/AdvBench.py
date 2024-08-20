import json
import pandas
from typing import List, Any


def get_adv_bench_behaviors(path="./resources/advbench/harmful_behaviors.csv") -> List[List[str]]:
    results = []
    df = pandas.read_csv(path)
    for index, row in df.iterrows():
        results.append([row[0], row[1]])
    return results


def get_adv_bench_behaviors_50(path="./resources/advbench/harmful_behaviors_50_i_gcg.json") -> List[List[str]]:
    """
    dataset used in Improved Techniques for Optimization-Based Jailbreaking on Large Language Models
    A subset of AdvBench with length 50.
    """
    results = []
    with open(path) as f:
        data = json.load(f)
    for item in data:
        results.append([item["behaviour"], item["target"]])
    return results
