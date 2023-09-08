# Python Built-in Modules
import logging
import math
import random
from typing import Dict, List, Tuple

# Third-Party Libraries
from torch.utils.data.sampler import Sampler

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig

logger = logging.getLogger(__name__)


class ERCDataSampler(Sampler):
    """
    Instead of simply consuming the dataset in random order, this sampler produces
    balanced batches containing at least two examples of each class. This is done
    to support the triplet loss, which requires at least 2 examples of each class.
    """

    indices_per_class: Dict[str, List[int]]
    balanced_size: int
    n_examples_per_class: int

    def __init__(self, labels: List[str], batch_size: int | None = None, config: ERCConfig | None = None):
        """
        Constructs an instance of ERCDataSampler, indexes the labels per class and
        calculates the balanced size of the dataset.

        :param labels: The ordered list of all labels in the dataset.
        :param batch_size: The batch_size if available, defaults to None
        """
        classes = sorted(set(labels))
        self.random = self._init_random(config)
        self.indices_per_class = self._init_indices_per_class(labels=labels, classes=classes)
        self.balanced_size = max(len(indices) for indices in self.indices_per_class.values()) * len(classes)
        self.n_examples_per_class = self._init_n_examples_per_class(batch_size=batch_size, classes=classes)

    @staticmethod
    def _init_random(config) -> random.Random:
        random_seed = config.classifier_seed if config else None
        if random_seed is None:
            logger.warn("Sampler: random seed is RECOMMENDED, but not provided, using default None")
        else:
            logger.warn(f"Sampler: random seed provided, '{random_seed}'")
        return random.Random(random_seed)

    @staticmethod
    def _init_indices_per_class(labels: List[str], classes: List[str]) -> Dict[str, List[int]]:
        ix: Dict[str, List[int]] = dict()
        for class_ in classes:
            ix[class_] = [i for i, current_label in enumerate(labels) if current_label == class_]
        return ix

    @staticmethod
    def _init_n_examples_per_class(classes: List[str], batch_size: int | None) -> int:
        if not batch_size or batch_size == 1:
            n_examples_per_class = 1
            logger.warn(f"Sampler: batch_size provided is '{batch_size}', won't support triplets, ok for test-sets")
        else:
            n_examples_per_class = max(2, math.ceil(batch_size / len(classes)))
            logger.warn(f"Sampler: batch_size={batch_size}, n_examples_per_class={n_examples_per_class}")
        return n_examples_per_class

    def __iter__(self):
        """
        Iterates through the classes and yields 2 indices per class, until the
        balanced size is reached.
        """
        iter_cursor = 0
        class_cursor = {k: 0 for k in self.indices_per_class.keys()}
        while iter_cursor < self.balanced_size:
            # every pass of all classes, randomise the order in which
            # the classes appear together, to allow triplet contrastive
            # learning to happen accross different class pairs, rather
            # than fixed ones.
            indices_of_shuffled_classes = self._randomise_class_indices_pairs()
            for class_, class_indices in indices_of_shuffled_classes:
                # yields as many items as required per class
                # which is optimised based on the batch size and
                # number of classes available in the datset
                for _ in range(self.n_examples_per_class):
                    try:
                        yield class_indices[class_cursor[class_] % len(class_indices)]
                    finally:
                        class_cursor[class_] += 1
                        iter_cursor += 1
                    if iter_cursor >= self.balanced_size:
                        return

    def _randomise_class_indices_pairs(self) -> List[Tuple[str, List[int]]]:
        """
        Changes the order in which the classes appear together, to allow triplet
        contrastive learning to happen accross different class pairs, rather than
        fixed ones.

        :return: List of indices per class.
        """
        pairs = list(self.indices_per_class.items())
        self.random.shuffle(pairs)
        return pairs

    def __len__(self):
        """
        Returns the balanced size of the dataset, which is
        num_classes * max(len(examples_per_class)), to allow all
        triplets to be formed against all classes in a balanced way.

        :return: Balanced size of the dataset.
        """
        return self.balanced_size
