# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig, ERCLossFunctions
from hlm12erc.modelling.erc_loss import ERCLoss


class TestCategoricalCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        config = ERCConfig(classifier_classes=["a", "b", "c"])
        self.loss = ERCLoss.resolve_type_from(ERCLossFunctions.CATEGORICAL_CROSS_ENTROPY)(config)

    def test_call(self):
        y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
        y_true = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        loss_value = self.loss(y_pred, y_true)
        expected_loss_value = torch.tensor(0.7355758547782898)
        self.assertAlmostEqual(loss_value.item(), expected_loss_value.item(), places=4)


class TestDiceCoefficientLoss(unittest.TestCase):
    def setUp(self):
        config = ERCConfig(classifier_classes=["a", "b", "c"])
        self.loss = ERCLoss.resolve_type_from(ERCLossFunctions.DICE_COEFFICIENT)(config)

    def test_call(self):
        y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
        y_true = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        loss_value = self.loss(y_pred, y_true)
        expected_loss_value = torch.tensor(0.5666666030883789)
        self.assertAlmostEqual(loss_value.item(), expected_loss_value.item(), places=4)
