# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch
from transformers import EvalPrediction

# My Packages and Modules
from hlm12erc.modelling import ERCConfig, ERCLossFunctions
from hlm12erc.training.erc_metric_calculator import ERCMetricCalculator


class TestERCMetricCalculator(unittest.TestCase):
    def setUp(self):
        loss_fn = ERCLossFunctions.CATEGORICAL_CROSS_ENTROPY
        config = ERCConfig(classifier_classes=["a", "b", "c"], classifier_loss_fn=loss_fn)
        self.metric_calculator = ERCMetricCalculator(config)
        self.eval_pred = EvalPrediction(
            predictions=torch.tensor(
                [
                    [0.1, 0.2, 0.7],
                    [0.3, 0.4, 0.3],
                    [0.3, 0.4, 0.3],
                    [0.3, 0.4, 0.3],
                ]
            ),
            label_ids=torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )

    def test_call_returns_loss_cce(self):
        output = self.metric_calculator(self.eval_pred)
        expected = 1.0417890548706055
        self.assertAlmostEqual(output["loss"], expected, places=5)

    def test_call_returns_loss_dice(self):
        metric_calculator = ERCMetricCalculator(
            ERCConfig(
                classifier_classes=["a", "b", "c"],
                classifier_loss_fn=ERCLossFunctions.DICE_COEFFICIENT,
            )
        )
        output = metric_calculator(self.eval_pred)
        expected = 0.7115942239761353
        self.assertAlmostEqual(output["loss"], expected, places=5)

    def test_call_returns_acc(self):
        output = self.metric_calculator(self.eval_pred)
        expected = 0.25
        self.assertAlmostEqual(output["acc"], expected, places=5)

    def test_call_returns_f1_weighted(self):
        output = self.metric_calculator(self.eval_pred)
        expected = 0.375
        self.assertAlmostEqual(output["f1_weighted"], expected, places=5)

    def test_determine_loss_has_no_grad_decorator(self):
        loss_fn = self.metric_calculator._determine_loss
        self.assertTrue(hasattr(loss_fn, "__wrapped__"))
