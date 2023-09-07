# Python Built-in Modules
import unittest
from typing import Any

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig
from hlm12erc.modelling.erc_loss import ERCTripletLoss
from hlm12erc.modelling.erc_output import ERCOutput
from hlm12erc.training.erc_trainer_job_triplet import ERCTrainerTripletJob


class TestERCTrainerTripletJob(unittest.TestCase):
    def setUp(self) -> None:
        config = ERCConfig(classifier_classes=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"])
        self.loss_fn = ERCTripletLoss(config=config)

    def test_tiplet_better_lower_than_worst(self):
        best = self._loss(t=["joy", "joy", "joy", "sadness", "sadness"], p=["joy", "joy", "joy", "sadness", "sadness"])
        worst = self._loss(t=["joy", "joy", "joy", "anger", "anger"], p=["anger", "neutral", "fear", "disgust", "fear"])
        self.assertLess(best, worst)

    def test_tiplet_better_lower_than_medium(self):
        best = self._loss(t=["joy", "joy", "joy", "sadness", "sadness"], p=["joy", "joy", "joy", "sadness", "sadness"])
        medium = self._loss(t=["joy", "joy", "joy", "anger", "anger"], p=["joy", "joy", "fear", "anger", "sadness"])
        self.assertLess(best, medium)

    def test_tiplet_medium_lower_than_worst(self):
        medium = self._loss(t=["joy", "joy", "joy", "anger", "anger"], p=["joy", "joy", "fear", "anger", "sadness"])
        worst = self._loss(t=["joy", "joy", "joy", "anger", "anger"], p=["anger", "neutral", "fear", "disgust", "fear"])
        self.assertLess(medium, worst)

    def _loss(self, t, p) -> float | torch.Tensor:
        return ERCTrainerTripletJob._compute_triplet_loss(
            outputs=_MockModel(return_labels=p)(labels=t),
            labels=_MockModel.tensor_for_label(labels=t),
            loss_fn=self.loss_fn,
        )


class _MockModel:
    def __init__(self, return_labels: list):
        self.return_labels = return_labels

    def __call__(self, labels: list, *args: Any, **kwds: Any) -> ERCOutput:
        unique_labels = set(self.return_labels + labels)
        unique_vectors = {lbl: torch.randn(7) for lbl in unique_labels}
        y_embeddings = torch.stack([unique_vectors[lbl] for lbl in self.return_labels])
        y_embeddings = torch.nn.functional.normalize(y_embeddings, p=2, dim=1)
        y_pred = _MockModel.tensor_for_label(self.return_labels, positive=0.4, negative=0.1)
        return ERCOutput(labels=y_pred, hidden_states=y_embeddings)

    @staticmethod
    def tensor_for_label(
        labels: list,
        positive: float = 1.0,
        negative: float = 0.0,
    ) -> torch.Tensor:
        data = [
            [
                positive if labels[i] == "anger" else negative,
                positive if labels[i] == "disgust" else negative,
                positive if labels[i] == "fear" else negative,
                positive if labels[i] == "joy" else negative,
                positive if labels[i] == "neutral" else negative,
                positive if labels[i] == "sadness" else negative,
                positive if labels[i] == "surprise" else negative,
            ]
            for i in range(len(labels))
        ]
        return torch.tensor(data)
