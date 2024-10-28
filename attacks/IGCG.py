import torch
from typing import Iterable, List
from .GCG import GCGAttack, get_embeddings, get_embedding_matrix
from models import BaseModel

__all__ = ["ImprovedGCG", "change_dataset_to_i_gcg_target", "IGCGOnlyEnumerate"]


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
        super(ImprovedGCG, self).__init__(*args, adv_string_init=adv_string_init, **kwargs)

    def attack(self):
        result = super(ImprovedGCG, self).attack()
        adv_suffix_now = result[len(self.prompt) + 1 :]
        self.adv_string_init = adv_suffix_now
        return result


class IGCGOnlyEnumerate(ImprovedGCG):
    def __init__(self, models: List[BaseModel], *args, adv_string_init="[ " * 20 + "[", **kwargs):
        topk = min([model.tokenizer.vocab_size for model in models])
        super(IGCGOnlyEnumerate, self).__init__(models, *args, topk=topk, adv_string_init=adv_string_init, **kwargs)

    def token_gradients(self, input_ids, input_slice, target_slice, loss_slice):
        """
        这里不计算梯度, 直接把梯度全部设置成0.
        input_ids: L
        所以本质上这个函数是没用的，只是方便封装，直接把梯度set为0，顺带return个topk方便可视化而已
        """
        model = self.models[0].model
        embed_weights = get_embedding_matrix(model)
        # 这里调用防御的forward
        logits = self.models[0](input_ids.unsqueeze(0)).logits
        targets = input_ids[target_slice]
        # 这里不计算梯度, 直接把梯度全部设置成0.
        grad = torch.zeros(input_ids[input_slice].shape[0], embed_weights.shape[0], device=model.device)

        # Compute topk
        prediction = torch.topk(logits[0, loss_slice, :], k=10, dim=1)[1]  # L, K
        position_table = prediction == targets.unsqueeze(1)
        topk = torch.max(position_table, dim=1)[1]
        topk = torch.where(position_table.sum(1) != 0, topk, float("inf"))
        return grad, topk
