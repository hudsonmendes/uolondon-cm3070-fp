# Python Built-in Modules
import math
import unittest

# Third-Party Libraries
import torch
from hypothesis import given
from hypothesis.strategies import integers

# My Packages and Modules
from hlm12erc.training.erc_data_sampler import ERCDataSampler


class TestERCDataSampler(unittest.TestCase):
    def setUp(self) -> None:
        classes = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        self.labels = [classes[i] for i in torch.randint(0, 7, (256,)).tolist()]

    def test_len_is_rebalanced(self):
        sampler = ERCDataSampler(labels=self.labels)
        self.assertGreater(len(sampler), len(self.labels))

    def test_batching_edgecases_batchsize_None(self):
        self._test_n_examples_per_class(batch_size=None, n=1, u=1)

    def test_batching_edgecases_batchsize_0(self):
        self._test_n_examples_per_class(batch_size=0, n=1, u=1)

    def test_batching_edgecases_batchsize_1(self):
        self._test_n_examples_per_class(batch_size=1, n=1, u=1)

    def test_batching_edgecases_batchsize_2(self):
        self._test_n_examples_per_class(batch_size=2, n=2, u=1)

    @given(batch_size=integers(min_value=3, max_value=128))
    def test_batching_generation(self, batch_size: int):
        expected_n_examples_per_class = max(2, math.ceil(batch_size / 7))
        expected_unique_labels = min(7, math.ceil(batch_size / expected_n_examples_per_class))
        self._test_n_examples_per_class(
            batch_size=batch_size,
            n=expected_n_examples_per_class,
            u=expected_unique_labels,
        )

    def _test_n_examples_per_class(self, batch_size: int | None, n: int, u: int):
        sampler = ERCDataSampler(labels=self.labels, batch_size=batch_size)
        iter = sampler.__iter__()
        actual = [next(iter) for _ in range(batch_size or 1)]
        self.assertEqual(n, sampler.n_examples_per_class)
        self.assertEqual(u, len(set([self.labels[i] for i in actual])))
