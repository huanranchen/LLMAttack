from typing import Iterable, List


def change_dataset_to_i_gcg_target(loader: Iterable or List[List[str]]) -> List[List[str]]:
    results = []
    for inputs, targets in loader:
        targets = targets.split("Sure,", 1)
        # targets = "Sure, " + "my output is harmful," + targets[1]
        targets = "Sure, " + "my output is helpful, " + targets[1]
        # targets = "Sure, " + "I will help you, " + targets[1]
        results.append([inputs, targets])
    return results
