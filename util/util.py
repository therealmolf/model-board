"""
    Utility Functions for total loss, accuracy, etc.
"""

import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import Counter


def compute_total_loss(model: t.nn.Module,
                       loader: t.utils.data.dataloader.DataLoader,
                       device=None):

    if device is None:
        device = t.device(device)

    model = model.eval()
    loss = 0
    examples = 0

    for idx, (features, labels) in enumerate(loader):

        features, labels = features.to(device), labels.to(device)

        with t.no_grad():
            logits = model(features)
            batch_loss = F.cross_entropy(features, labels, reduction="sum")

        loss += batch_loss.item()
        examples += logits.shape[0]

    return loss / examples


def compute_accuracy(model: t.nn.Module,
                     loader: t.utils.data.dataloader.DataLoader):

    model = model.eval()

    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(loader):

        with t.no_grad():
            logits = model(features)

        preds = t.argmax(logits, dim=1)

        compare = labels == preds
        correct += t.sum(compare)
        total_examples += len(compare)

    return correct / total_examples


def count_classes(loader: DataLoader):
    """
        Count and sort the number of labels/classes from a DataLoader
    """

    counter = Counter()

    for _, labels in loader:
        counter.update(labels.tolist())

    return (sorted(counter.items()), counter)
