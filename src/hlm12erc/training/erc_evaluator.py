# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling import ERCModel

# Local Folders
from .meld_dataset import MeldDataset


class ERCEvaluator:
    def __init__(self, model: ERCModel) -> None:
        self.model = model

    def evaluate(self, ds: MeldDataset) -> None:
        """
        Evaluates the model on the given dataset.

        :param ds: Dataset to be evaluated.
        """
        self.model.eval()
        with torch.no_grad():
            for batch in ds:
                self.model(batch)
