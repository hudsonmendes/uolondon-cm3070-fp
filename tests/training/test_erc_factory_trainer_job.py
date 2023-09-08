# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch
import transformers
from hypothesis import given
from hypothesis.strategies import integers

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig
from hlm12erc.training.erc_factory_trainer_job import ERCTrainerJobFactory


class TestERCTrainerJobFactory(unittest.TestCase):
    @given(integers(min_value=1, max_value=100))
    def test_create_with_earlystopping_on(self, patience: int):
        actual = self._create_subject(dict(classifier_early_stopping_patience=patience))
        self.assertTrue(
            any([isinstance(c, transformers.EarlyStoppingCallback) for c in actual.callback_handler.callbacks])
        )

    @given(integers(min_value=-100, max_value=0))
    def test_create_with_earlystopping_off_with_negative_or_zero_patience(self, patience: int):
        actual = self._create_subject(dict(classifier_early_stopping_patience=patience))
        self.assertTrue(
            not any([isinstance(c, transformers.EarlyStoppingCallback) for c in actual.callback_handler.callbacks])
        )

    def test_create_with_earlystopping_off_with_null_patience(self):
        actual = self._create_subject(dict(classifier_early_stopping_patience=None))
        self.assertTrue(
            not any([isinstance(c, transformers.EarlyStoppingCallback) for c in actual.callback_handler.callbacks])
        )

    def _create_subject(self, modified_config: dict):
        configs = dict(classifier_classes=["a", "b", "c"], **modified_config)
        config = ERCConfig(**configs)
        factory = ERCTrainerJobFactory(config)
        model = torch.nn.Linear(1, 1)
        actual = factory.create(
            model=model,
            train_dataset=None,
            eval_dataset=None,
            training_args=None,
            label_encoder=None,
        )
        return actual
