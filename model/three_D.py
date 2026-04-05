import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F


class DynamicTaskPrioritization:
    def __init__(self, num_tasks=5, alpha=0.1):
        self.alpha = alpha
        self.task_weights = torch.ones(num_tasks, requires_grad=False)
        self.prev_kpis = torch.ones(num_tasks)

    def update_weights(self, losses, init=False):
        # current_kpis = 1 / (1 - losses + 1e-6)
        for i in range(len(losses)):
            # self.task_weights[i] = self.alpha * current_kpis[i] + (1 - self.alpha) * self.task_weights[i]
            self.task_weights[i] = self.alpha * losses[i] + (1 - self.alpha) * self.task_weights[i]

            # self.task_weights[i] = self.task_weights[i] * self.task_weights[i] * self.task_weights[i]
        # 归一化权重，确保稳定性
        if init:
            for i in range(len(losses)):
                self.task_weights[i] = 1

        self.task_weights /= self.task_weights.sum()

