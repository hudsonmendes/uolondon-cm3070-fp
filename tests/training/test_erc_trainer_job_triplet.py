# Python Built-in Modules
import unittest
from typing import Any

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_output import ERCOutput
from hlm12erc.training.erc_trainer_job_triplet import ERCTrainerTripletJob


class TestERCTrainerTripletJob(unittest.TestCase):
    def test_tiplet_better_lower_than_worst(self):
        best = ERCTrainerTripletJob._compute_triplet_loss(
            outputs=_MockModel(results=["joy", "joy", "sadness", "sadness"])(),
            labels=_MockModel.label_tensor(labels=["joy", "joy", "sadness", "sadness"]),
        )
        worst = ERCTrainerTripletJob._compute_triplet_loss(
            outputs=_MockModel(results=["anger", "anger", "joy", "joy"])(),
            labels=_MockModel.label_tensor(labels=["joy", "joy", "sadness", "sadness"]),
        )
        self.assertLess(best, worst)

    def test_tiplet_better_lower_than_medium(self):
        best = ERCTrainerTripletJob._compute_triplet_loss(
            outputs=_MockModel(results=["joy", "joy", "sadness", "sadness"])(),
            labels=_MockModel.label_tensor(labels=["joy", "joy", "sadness", "sadness"]),
        )
        medium = ERCTrainerTripletJob._compute_triplet_loss(
            outputs=_MockModel(results=["joy", "joy", "sadness", "neutral"])(),
            labels=_MockModel.label_tensor(labels=["joy", "joy", "sadness", "sadness"]),
        )
        self.assertLess(best, medium)

    def test_tiplet_medium_lower_than_worst(self):
        medium = ERCTrainerTripletJob._compute_triplet_loss(
            outputs=_MockModel(results=["joy", "joy", "sadness", "neutral"])(),
            labels=_MockModel.label_tensor(labels=["joy", "joy", "sadness", "sadness"]),
        )
        worst = ERCTrainerTripletJob._compute_triplet_loss(
            outputs=_MockModel(results=["joy", "joy", "sadness", "neutral"])(),
            labels=_MockModel.label_tensor(labels=["joy", "joy", "sadness", "sadness"]),
        )
        self.assertLess(medium, worst)


class _MockModel:
    @staticmethod
    def label_tensor(labels: list, positive: float = 1.0, negative: float = 0.0):
        return torch.tensor(
            [
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
        )

    def __init__(self, results: list):
        self.batch_size = len(results)
        self.labels = _MockModel.label_tensor(results, positive=0.4, negative=0.1)

    @torch.no_grad()
    def __call__(self, *args: Any, **kwds: Any) -> ERCOutput:
        mock_embeddings = torch.rand((4, 256))
        mock_labels = torch.tensor(self.labels)
        return ERCOutput(
            labels=mock_labels,
            hidden_states=torch.nn.functional.normalize(mock_embeddings, p=2, dim=1),
        )
