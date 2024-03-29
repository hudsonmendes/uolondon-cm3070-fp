# Python Built-in Modules
from typing import Any, Callable, Dict, Tuple, Union

# Third-Party Libraries
import torch
import transformers
from torch.utils.data import DataLoader

# My Packages and Modules
from hlm12erc.modelling import ERCConfig, ERCOutput, ERCTripletLoss

# Local Folders
from .erc_data_collator import ERCDataCollator
from .erc_data_sampler import ERCDataSampler
from .meld_dataset import MeldDataset


class ERCTrainerTripletJob(transformers.Trainer):
    """
    Overrides the huggingface Trainer class to change the:
    (a) `compute_loss` that works on triplets
    (b) `compute_metrics` that calculates the accuracy, f1, precision and recall
    (c) `get_train_dataloader` that batches at least one example of each class

    The triplet loss calculated based on the embeddings of the anchor, positive
    and negative examples is added to the mean dice loss of the individual examples.
    """

    custom_metric_computation: Callable[[transformers.EvalPrediction], Dict[str, Any]] | None = None

    def __init__(
        self,
        config: ERCConfig,
        compute_metrics: transformers.EvalPrediction | None = None,
        *args,
        **kwargs,
    ):
        """
        Constructs an instance of ERCTrainerTripletJob.

        :param config: ERCConfig with hyperparams
        :param compute_metrics: Callable object to calculate the metrics.
        """
        super(ERCTrainerTripletJob, self).__init__(compute_metrics=compute_metrics, *args, **kwargs)
        self.config = config
        self.custom_metric_computation = compute_metrics
        self.triplet_loss = ERCTripletLoss(config=config)

    def compute_loss(self, model, inputs, return_outputs=False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Computes and sums the classification loss and the triplet loss

        :param model: Model to compute the loss for.
        :param inputs: Inputs to the model.
        :param return_outputs: Whether to return the outputs of the model.
        :return: Loss of the model.
        """
        # get hold of labels, required to calculate metrics & triplet loss
        labels = inputs.get(ERCDataCollator.LABEL_NAME)

        # regardless of whether we are training using the triplet loss or not,
        # we must still compute classifier metrics. The reason for that is because
        # we must be able to compare the metrics of this approach with the metrics
        # of the other approaches.
        outputs = model(**inputs)
        total_loss = classifier_loss = outputs.loss
        classifier_metrics = dict()

        if labels is not None:
            classifier_metrics = self._compute_metrics(labels, outputs, loss=classifier_loss.item())

            # if we are training using the triplet loss, we must compute the triplet loss
            if self.is_in_train:
                triplet_loss = self._compute_triplet_loss(outputs=outputs, labels=labels, loss_fn=self.triplet_loss)
                total_loss += triplet_loss
                classifier_metrics.update(dict(triplet_loss=triplet_loss.item(), total_loss=total_loss.item()))

        if classifier_metrics:
            self.log(classifier_metrics)

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_metrics(self, labels: torch.Tensor, outputs: ERCOutput, loss: float) -> Dict[str, Any]:
        """
        Runs the `custom_metric_computation` procedure to calculate the metrics
        that we want to track, and then updates the metrics dictionary with the
        loss and the custom metrics.

        :param labels: Labels of the inputs.
        :param outputs: Outputs of the model.
        :param loss: Loss of the model, float format (tensor.item())
        :return: Dictionary containing the metrics.
        """
        metrics = dict(loss=loss)
        if self.custom_metric_computation is not None:
            eval_pred = transformers.EvalPrediction(predictions=outputs.labels, label_ids=labels)
            custom_metrics = self.custom_metric_computation(eval_pred)
            metrics.update(custom_metrics)
        return metrics

    @staticmethod
    def _compute_triplet_loss(outputs: ERCOutput, labels: torch.Tensor, loss_fn: ERCTripletLoss) -> torch.Tensor:
        """
        Calculates the triplet loss using the TripletLoss.

        :param outputs: Outputs of the model.
        :param labels: Labels of the inputs.
        :param loss_fn: TripletLoss object.
        """
        # ensures that there are at least 2 examples of each class present as label
        # otherwise the triplet loss can't be effectively calculated
        argmax_labels = labels.detach().argmax(dim=1)
        unique_labels, count_per_label = argmax_labels.unique(return_counts=True)
        if torch.any(count_per_label > 1):
            # we segregate the embeddings based on the labels
            # so that we can create anchors, positives and negative exampels
            class_embeds = []
            for label_id in unique_labels:
                indices = argmax_labels == label_id
                class_embeds.append(outputs.hidden_states[indices])

            # the tripplet losses are calculated and acculuated.
            losses = []
            for i in range(len(class_embeds)):
                # we can only process classes with at least 2 examples in the batch
                # because we need at least 1 positive and 1 negative example. however
                # classes without a 2nd example can still be processed as negatives
                if class_embeds[i].shape[0] > 1:
                    after_i = i + 1
                    anchor = class_embeds[i][0]
                    positives = class_embeds[i][1:]
                    negatives_array = class_embeds[:i] + class_embeds[after_i:]
                    if len(negatives_array) > 0:
                        negatives = torch.cat(negatives_array)
                        loss = loss_fn(anchor=anchor, positives=positives, negatives=negatives)
                        losses.append(loss)

            # once we have all positives and all negatives for the batch
            # we use an adaptation of the SimCSE loss function to calculate the loss
            # epsilon is added for numerical stability.
            if losses:
                return torch.mean(torch.stack(losses))

        # for edge cases where the triplet loss cannot be calculated,
        # simply return 0.0
        return torch.tensor(0.0)

    def get_train_dataloader(self) -> DataLoader:
        """
        Overrides the get_train_dataloader method to use the ERCDataSampler.

        :return: DataLoader for the dataset.
        """
        assert isinstance(self.train_dataset, MeldDataset)
        sampler = ERCDataSampler(self.train_dataset.labels, batch_size=self.args.train_batch_size, config=self.config)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
