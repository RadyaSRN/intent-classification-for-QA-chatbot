import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, input_ids=None, labels=None):
        logits = self.classifier(input_ids)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)