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
        expected_loss_value = 0.7355758547782898
        self.assertAlmostEqual(loss_value.item(), expected_loss_value, places=4)


class TestDiceCoefficientLoss(unittest.TestCase):
    def setUp(self):
        config = ERCConfig(classifier_classes=["a", "b", "c"])
        self.loss = ERCLoss.resolve_type_from(ERCLossFunctions.DICE_COEFFICIENT)(config)

    def test_call(self):
        y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
        y_true = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        loss_value = self.loss(y_pred, y_true)
        expected_loss_value = 0.6976743936538696
        self.assertAlmostEqual(loss_value.item(), expected_loss_value, places=4)


class TestFocalMutiClassLogLoss(unittest.TestCase):
    def setUp(self):
        config = ERCConfig(classifier_classes=["a", "b", "c", "d"], losses_focal_alpha=[0.25, 0.25, 0.25, 0.25])
        self.loss = ERCLoss.resolve_type_from(ERCLossFunctions.FOCAL_MULTI_CLASS_LOG)(config)

    def test_call(self):
        y_pred = torch.tensor([[0.1, 0.8, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05], [0.1, 0.6, 0.1, 0.2]])
        y_true = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        loss_value = self.loss(y_pred, y_true)
        expected_loss_value = 0.31159278750419617
        self.assertAlmostEqual(loss_value.item(), expected_loss_value, places=4)
