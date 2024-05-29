import torch
from abc import abstractmethod
from models.BaseModel import BaseModel
from typing import List


class BaseAttacker:
    def __init__(self, models: List[BaseModel]):
        self.models = models

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)

    @abstractmethod
    def attack(self, *args, **kwargs):
        """
        :param args: a suffix string
        :param kwargs: configurations
        :return: an optimized suffix string
        """
        pass
