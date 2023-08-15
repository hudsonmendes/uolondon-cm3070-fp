# Python Built-in Modules
from typing import Any, Dict, List, Tuple

# Third-Party Libraries
import torch
from sklearn.metrics import classification_report
from tqdm.auto import trange

# My Packages and Modules
from hlm12erc.modelling import ERCLabelEncoder, ERCModel

# Local Folders
from .erc_data_collator import ERCDataCollator
from .meld_dataset import MeldDataset


class ERCEvaluator:
    model: ERCModel
    label_encoder: ERCLabelEncoder
    data_collator: ERCDataCollator

    def __init__(self, model: ERCModel) -> None:
        self.model = model
        self.label_encoder = model.label_encoder
        self.data_collator = ERCDataCollator(label_encoder=model.label_encoder, config=model.config)

    def evaluate(
        self,
        dataset: MeldDataset,
        device: torch.device | None = None,
        batch_size: int = 4,
    ) -> Dict[str, Any]:
        if device is not None:
            self.model.to(device)

        emotions = self.label_encoder.classes
        labels, preds = None, None
        with torch.no_grad():
            labels, preds = self._collect_labels_and_preds(dataset, device, batch_size)

        return self._report_on_results(emotions, labels, preds)

    def _collect_labels_and_preds(self, dataset, device, batch_size) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Collects the one hot encodings of the true labels and the prediction softmax distributions,
        stack them and run an argmax (on axis 1) to get the integers representing the labels.

        :param dataset: The dataset to evaluate on, usually the test split.
        :param device: The device to run the data collation on
        :param batch_size: The size of the batches that will be run against the model
        :return: A tuple of two lists, the first being the true labels and the second being the predicted labels
        """
        y_true, y_pred = [], []
        for start in trange(0, len(dataset), batch_size, desc="evaluating"):
            end = start + batch_size
            batch_records = dataset[start:end]
            batch_collated = self.data_collator(batch_records, device=device)
            y_true.extend(batch_collated["y_true"])
            y_pred.extend(self.model(**batch_collated).labels)
        labels = torch.stack(y_true, dim=0).argmax(dim=1).tolist()
        preds = torch.stack(y_pred, dim=0).argmax(dim=1).tolist()
        return labels, preds

    def _report_on_results(self, emotions, labels, preds) -> Dict[str, Any]:
        """
        Prints the classification report and returns the report dictionary.

        :param emotions: The list of emotions
        :param labels: The true labels
        :param preds: The predicted labels
        :return: The classification report dictionary
        """
        report_dict = classification_report(y_true=labels, y_pred=preds, target_names=emotions, output_dict=True)
        print(classification_report(y_true=labels, y_pred=preds, target_names=emotions))
        return report_dict
