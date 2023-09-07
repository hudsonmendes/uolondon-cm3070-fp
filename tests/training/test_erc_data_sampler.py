# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.training.erc_data_sampler import ERCDataSampler


class TestERCDataSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = torch.randint(0, 6, (31,)).tolist()
        self.sampler = ERCDataSampler(labels=self.labels)

    def test_len_is_rebalanced(self):
        self.assertGreater(len(self.sampler), len(self.labels))

    def test_batch_of_4_has_2_pairs(self):
        iter = self.sampler.__iter__()
        actual = []
        actual.append(next(iter))
        actual.append(next(iter))
        actual.append(next(iter))
        actual.append(next(iter))
        self.assertListEqual([0, 0, 1, 1], [self.labels[i] for i in actual])
