from typing import Iterable, List
from .GCG import GCGAttack

__all__ = ["ImprovedGCG", "change_dataset_to_i_gcg_target"]


def change_dataset_to_i_gcg_target(loader: Iterable or List[List[str]]) -> List[List[str]]:
    results = []
    for inputs, targets in loader:
        targets = targets.split("Sure,", 1)
        # targets = "Sure, " + "my output is harmful," + targets[1]
        targets = "Sure, " + "my output is helpful, " + targets[1]
        # targets = "Sure, " + "I will help you, " + targets[1]
        results.append([inputs, targets])
    return results


class ImprovedGCG(GCGAttack):
    def __init__(self, *args, adv_string_init="[ " * 20, **kwargs):
        super(GCGAttack).__init__(*args, adv_string_init=adv_string_init, **kwargs)

    def attack(self):
        result = super(ImprovedGCG, self).attack()
        adv_suffix_now = result[len(self.prompt) + 1 :]
        self.adv_string_init = adv_suffix_now
        return result
