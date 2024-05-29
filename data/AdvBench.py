import pandas
from typing import List, Any


def get_adv_bench_behaviors(path="./resources/advbench/harmful_behaviors.csv") -> List[List[Any]]:
    results = []
    df = pandas.read_csv(path)
    for index, row in df.iterrows():
        results.append([row[0], row[1]])
    return results
