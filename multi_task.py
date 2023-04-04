import torch.nn as nn


class MultiTaskModel(nn.Module):
    def __init__(self, base_net, task_heads, device):
        super().__init__()
        self.base_net = base_net
        self.task_heads = nn.ModuleList(task_heads)
        self.device = device

    def forward(self, x):

        base_output = self.base_net(x)
        # Forward pass through task-specific heads
        task_outputs = [head(base_output) for head in self.task_heads]

        return task_outputs

    def predict(self, x):

        base_output = self.base_net(x)
        # Forward pass through task-specific heads
        task_outputs = [head.predict(base_output) for head in self.task_heads]

        return task_outputs

    def accuracy(self, predictions, targets):
        accuracies = [head.accuracy(prediction, target) for head, prediction, target in zip(
            self.task_heads, predictions, targets)]
        return accuracies

    def recall(self, predictions, targets):
        recalls = [head.recall(prediction, target) for head, prediction, target in zip(
            self.task_heads, predictions, targets)]
        return recalls
    
    def precision(self, predictions, targets):
        precisions = [head.precision(prediction, target) for head, prediction, target in zip(
            self.task_heads, predictions, targets)]
        return precisions