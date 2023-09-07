# Python Built-in Modules
from typing import Dict, List

# Third-Party Libraries
from torch.utils.data.sampler import Sampler


class ERCDataSampler(Sampler):
    """
    Instead of simply consuming the dataset in random order, this sampler produces
    balanced batches containing at least two examples of each class. This is done
    to support the triplet loss, which requires at least 2 examples of each class.
    """

    labels: List[str]
    indices_by_label: Dict[str, List[int]]
    balanced_size: int

    def __init__(self, labels):
        """
        Constructs an instance of ERCDataSampler, indexes the labels per class and
        calculates the balanced size of the dataset.

        :param labels: List of labels.
        """
        self.labels = labels
        self.indices_by_label = {}
        self.balanced_size = len(labels)

        unique_labels = sorted(set(labels))
        for label in unique_labels:
            self.indices_by_label[label] = [i for i, current_label in enumerate(labels) if current_label == label]
        self.balanced_size = max(len(indices) for indices in self.indices_by_label.values()) * len(unique_labels)

    def __iter__(self):
        """
        Iterates through the classes and yields 2 indices per class, until the
        balanced size is reached.
        """
        cursor = 0
        while cursor < self.balanced_size:
            for label_indices in self.indices_by_label.values():
                # yields the first item of the pair
                try:
                    yield label_indices[cursor % len(label_indices)]
                finally:
                    cursor += 1

                # check if we should stop
                if cursor >= self.balanced_size:
                    return

                # yields the second item of the pair
                try:
                    yield label_indices[cursor % len(label_indices)]
                finally:
                    cursor += 1

                # check if we should stop
                if cursor >= self.balanced_size:
                    return

    def __len__(self):
        return self.balanced_size
