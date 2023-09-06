# Python Built-in Modules
from typing import Any, Callable, Dict, Tuple, Union

# Third-Party Libraries
import torch
import transformers

# My Packages and Modules
from hlm12erc.modelling import ERCOutput

# Local Folders
from .erc_data_collator import ERCDataCollator


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
        compute_metrics: transformers.EvalPrediction | None = None,
        *args,
        **kwargs,
    ):
        """
        Constructs a Custom Trainer keeping the `compute_metrics` object to
        be used to calculate the metrics within the `compute_loss` function.

        :param compute_metrics: Callable object to calculate the metrics.
        """
        super(ERCTrainerTripletJob, self).__init__(compute_metrics=compute_metrics, *args, **kwargs)
        self.custom_metric_computation = compute_metrics

    def compute_loss(self, model, inputs, return_outputs=False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Computes the combined classification loss and the triplet loss.
        The classification loss is the mean CrossEntropy, Dice or Focal loss of the examples.
        The triplet loss is based on the SimCSE paper.

        To calculate the triplet loss, it's vital that the data loader produces
        a batch containing 8 elements, where the first 7 elements are the examples
        of the 7 classes, and the last element is the example of the anchor class.

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
            triplet_loss = self._compute_triplet_loss(outputs, labels)
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
    def _compute_triplet_loss(outputs, labels) -> torch.Tensor:
        """
        Calculate the triplet loss is based on the SimCSE paper (see reference)
        which is given by the following equation.
        >>> triplet_loss = -log(
            exp(sim(anchor, positive)) /
            sum(exp(sim(anchor, positive)) + exp(sim(anchor, negative)))
        )

        Reference:
        >>> Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple
        ... Contrastive Learning of Sentence Embeddings. In EMNLP 2021 - 2021
        ... Conference on Empirical Methods in Natural Language Processing,
        ... Proceedings (EMNLP 2021 - 2021 Conference on Empirical Methods in
        ... Natural Language Processing, Proceedings), Association for
        ... Computational Linguistics (ACL), 6894–6910.
        """
        # we must ensure that we have at least examples from 2 classes
        # and check wether there's at least 2 examples per class.
        argmax_labels = labels.detach().argmax(dim=1)
        unique_labels = argmax_labels.unique()

        # each one of the rows of the outputs_hidden state is of a different class.
        # we now collect each embedding as separate tensors so that we can use them
        # as anchor, positive and negative examples.
        class_embeds = []
        for label_id in unique_labels:
            class_embeds.append(outputs.hidden_states[argmax_labels == label_id])

        # we now calculate the similarity between the anchor and the positive
        # examples, and the anchor and the negative examples. only examples
        # with a positive can have their negatives calculated.
        #
        # differently to the SimCSE implementation, the present approach does
        # not use a dual encoder. Instead it uses simply a sliding window of
        # positives to create the denominator of the triplet loss.
        losses = []
        for i in range(len(class_embeds)):
            after_i = i + 1
            for j in range(class_embeds[i].size(dim=0) - 1):
                after_j, next_after_j = j + 1, j + 2
                anchor = class_embeds[i][j]
                positives = torch.cat((class_embeds[i][:after_j], class_embeds[i][next_after_j:]))
                negatives = torch.cat((class_embeds[:i] + class_embeds[after_i:]))
                similars = torch.cosine_similarity(anchor, positives, dim=1)
                disimilars = torch.cosine_similarity(anchor, negatives, dim=1)
                similarities = torch.cat((similars, disimilars))
                ratio = torch.sum(similars) / torch.sum(similarities)
                loss = -torch.log(ratio)
                losses.append(loss)

        # once we have all positives and all negatives for the batch
        # we use an adaptation of the SimCSE loss function to calculate the loss
        # epsilon is added for numerical stability.
        return torch.sum(torch.cat(losses))
