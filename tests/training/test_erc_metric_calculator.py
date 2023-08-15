# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction

# My Packages and Modules
from hlm12erc.training.erc_metric_calculator import ERCMetricCalculator


class TestERCMetricCalculator(unittest.TestCase):
    def setUp(self):
        self.metric_calculator = ERCMetricCalculator(classifier_loss_fn="cce")
        self.eval_pred = EvalPrediction(
            predictions=torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]),
            label_ids=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        )

    def test_call_returns_loss(self):
        output = self.metric_calculator(self.eval_pred)
        expected = 0.9505091905
        self.assertAlmostEqual(output["loss"], expected, places=5)

    def test_call_returns_acc(self):
        output = self.metric_calculator(self.eval_pred)
        expected = 0.5
        self.assertAlmostEqual(output["acc"], expected, places=5)

    def test_call_returns_f1_weighted(self):
        output = self.metric_calculator(self.eval_pred)
        expected = 0.5
        self.assertAlmostEqual(output["f1_weighted"], expected, places=5)
