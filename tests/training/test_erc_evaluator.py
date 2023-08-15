# Python Built-in Modules
import itertools
import pathlib
import unittest

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig
from hlm12erc.modelling.erc_label_encoder import ERCLabelEncoder
from hlm12erc.modelling.erc_model import ERCModel
from hlm12erc.training.erc_evaluator import ERCEvaluator
from hlm12erc.training.meld_dataset import MeldDataset


class TestERCEvaluator(unittest.TestCase):
    def setUp(self):
        self.device = None
        self.config = ERCConfig(classifier_classes=["neutral", "joy"])
        self.label_encoder = ERCLabelEncoder(classes=self.config.classifier_classes)
        self.model = ERCModel(config=self.config, label_encoder=self.label_encoder)
        self.evaluator = ERCEvaluator(model=self.model)
        self.dataset = MeldDataset(pathlib.Path("tests/fixtures/sample.csv"))

    def test_evaluate_returns_report_dict(self):
        report_dict = self.evaluator.evaluate(dataset=self.dataset, device=self.device, batch_size=4)
        self.assertIsInstance(report_dict, dict)

    def test_evaluate_returns_report_dict_with_correct_keys(self):
        report_dict = self.evaluator.evaluate(dataset=self.dataset, device=self.device, batch_size=4)
        expected_keys = ["joy", "neutral", "accuracy", "macro avg", "weighted avg"]
        self.assertListEqual(list(report_dict.keys()), expected_keys)

    def test_evaluate_returns_report_dict_with_correct_metrics_for_brokendown_metrics(self):
        report_dict = self.evaluator.evaluate(dataset=self.dataset, device=self.device, batch_size=4)
        metric_groups = self.label_encoder.classes + ["macro avg", "weighted avg"]
        expected_keys = ["precision", "recall", "f1-score", "support"]
        for class_name in metric_groups:
            self.assertListEqual(list(report_dict[class_name].keys()), expected_keys)

    def test_evaluate_returns_report_dict_with_single_metric_for_acc(self):
        report_dict = self.evaluator.evaluate(dataset=self.dataset, device=self.device, batch_size=4)
        self.assertIsInstance(report_dict["accuracy"], float)

    def test_evaluate_returns_report_dict_with_correct_breakdown(self):
        report_dict = self.evaluator.evaluate(dataset=self.dataset, device=self.device, batch_size=4)
        metric_groups = self.label_encoder.classes + ["macro avg", "weighted avg"]
        breakdown_metrics = ["precision", "recall", "f1-score"]
        for metric_group, breakdown_item in itertools.product(metric_groups, breakdown_metrics):
            self.assertIsInstance(report_dict[metric_group][breakdown_item], float)

    def test_evaluate_returns_report_dict_with_correct_support_for_classes(self):
        report_dict = self.evaluator.evaluate(dataset=self.dataset, device=self.device, batch_size=4)
        for metric_group in self.label_encoder.classes:
            self.assertEqual(report_dict[metric_group]["support"], 1)

    def test_evaluate_returns_report_dict_with_correct_support_for_aggregated(self):
        report_dict = self.evaluator.evaluate(dataset=self.dataset, device=self.device, batch_size=4)
        for metric_group in ["macro avg", "weighted avg"]:
            self.assertEqual(report_dict[metric_group]["support"], 2)
