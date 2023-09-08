# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig, ERCLossFunctions
from hlm12erc.modelling.erc_loss import ERCLoss, ERCTripletLoss


class TestCategoricalCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        config = ERCConfig(classifier_classes=["a", "b", "c"])
        self.loss = ERCLoss.resolve_type_from(ERCLossFunctions.CROSSENTROPY)(config)

    def test_call(self):
        y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
        y_true = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        loss_value = self.loss(y_pred, y_true)
        expected_loss_value = 0.7355758547782898
        self.assertAlmostEqual(loss_value.item(), expected_loss_value, places=4)


class TestDiceCoefficientLoss(unittest.TestCase):
    def setUp(self):
        config = ERCConfig(classifier_classes=["a", "b", "c"])
        self.loss = ERCLoss.resolve_type_from(ERCLossFunctions.DICE)(config)

    def test_call(self):
        y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
        y_true = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        loss_value = self.loss(y_pred, y_true)
        expected_loss_value = 0.6976743936538696
        self.assertAlmostEqual(loss_value.item(), expected_loss_value, places=4)


class TestFocalMutiClassLogLoss(unittest.TestCase):
    def setUp(self):
        config = ERCConfig(classifier_classes=["a", "b", "c", "d"], losses_focal_alpha=[0.25, 0.25, 0.25, 0.25])
        self.loss = ERCLoss.resolve_type_from(ERCLossFunctions.FOCAL)(config)

    def test_call(self):
        y_pred = torch.tensor([[0.1, 0.8, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05], [0.1, 0.6, 0.1, 0.2]])
        y_true = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        loss_value = self.loss(y_pred, y_true)
        expected_loss_value = 0.31159278750419617
        self.assertAlmostEqual(loss_value.item(), expected_loss_value, places=4)


class TestTripletLoss(unittest.TestCase):
    def setUp(self):
        self.anchor = self.positive = torch.tensor(
            [
                [0.1, 0.8, 0.05, 0.05],
                [0.1, 0.8, 0.05, 0.05],
                [0.1, 0.6, 0.1, 0.2],
            ]
        )
        self.negative = torch.rand(self.anchor.shape)
        self.loss = ERCTripletLoss(config=ERCConfig(classifier_classes=["a", "b", "c"]))

    def test_call_best_better_than_indiferent_positive_or_reversed(self):
        loss_xs = self.loss(anchor=self.anchor, positives=self.positive, negatives=self.negative)
        loss_md = self.loss(anchor=self.anchor, positives=self.positive, negatives=self.positive)
        loss_lg = self.loss(anchor=self.anchor, positives=self.negative, negatives=self.positive)
        self.assertLess(loss_xs, loss_md)
        self.assertLess(loss_xs, loss_lg)

    def test_call_best_better_than_indiferent_negative_or_reversed(self):
        loss_xs = self.loss(anchor=self.anchor, positives=self.positive, negatives=self.negative)
        loss_md = self.loss(anchor=self.anchor, positives=self.negative, negatives=self.negative)
        loss_lg = self.loss(anchor=self.anchor, positives=self.negative, negatives=self.positive)
        self.assertLess(loss_xs, loss_md)
        self.assertLess(loss_xs, loss_lg)
