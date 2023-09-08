# Python Built-in Modules
import pathlib
import unittest

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig
from hlm12erc.training.erc_factory_trainer_ta import ERCTrainerJobTrainingArgsFactory


class TestERCTrainerJobTrainingArgsFactory(unittest.TestCase):
    def test_create_best_metric_loss(self):
        actual = self._test_create(dict(classifier_metric_for_best_model="loss"))
        self.assertEqual("loss", actual.metric_for_best_model)

    def test_create_best_metric_f1_weighted(self):
        actual = self._test_create(dict(classifier_metric_for_best_model="f1_weighted"))
        self.assertEqual("f1_weighted", actual.metric_for_best_model)

    def _test_create(self, modified_config: dict):
        configs = dict(classifier_classes=["a", "b", "c"], **modified_config)
        config = ERCConfig(**configs)
        factory = ERCTrainerJobTrainingArgsFactory(config)
        return factory.create(
            n_epochs=1,
            batch_size=1,
            model_name="model_name",
            workspace=pathlib.Path("/tmp"),
        )
