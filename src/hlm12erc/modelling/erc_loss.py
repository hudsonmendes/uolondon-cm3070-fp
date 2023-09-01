# Python Built-in Modules
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type

# Third-Party Libraries
import torch
import torch.nn.functional as F

# Local Folders
from .erc_config import ERCConfig, ERCLossFunctions


class ERCLoss(ABC, torch.nn.Module):
    """
    Defines the contract of loss functions for ERC models.

    Example:
        >>> loss = ERCLoss.resolve_type_from("cce")()
        >>> loss(y_true=y_true, y_pred=y_pred)
    """

    config: ERCConfig

    def __init__(self, config: ERCConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        When implemented, this method should calculate and return the loss
        given the predicted and true labels.
        """
        raise NotImplementedError("ERCLoss is an abstract class.")

    @staticmethod
    def resolve_type_from(expression: str) -> Type["ERCLoss"]:
        """
        Resolve a ERC Loss class from a string expression.
        Some loss function configurations are a combination of a raw loss &
        another contrastive loss, e.g. "cce+triplet".

        In this case, the contrastive loss is ignored for the instantiation,
        because the contrastive loss is processed at the trainer level and
        has implications to the data loading process.
        """
        if expression == ERCLossFunctions.CROSSENTROPY or expression == ERCLossFunctions.CROSSENTROPY_PLUS_TRIPLET:
            return CategoricalCrossEntropyLoss
        elif expression == ERCLossFunctions.DICE or expression == ERCLossFunctions.DICE_PLUS_TRIPET:
            return DiceCoefficientLoss
        elif expression == ERCLossFunctions.FOCAL or expression == ERCLossFunctions.FOCAL_PLUS_TRIPLET:
            return FocalMutiClassLogLoss
        else:
            raise ValueError(f"Unknown ERC Loss type {expression}")


class CategoricalCrossEntropyLoss(ERCLoss):
    """
    Categorical Cross Entropy Loss function for ERC models.
    """

    def __init__(self, config: ERCConfig):
        super().__init__(config=config)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate and return the loss given the predicted and true labels.
        """
        return self.loss(y_pred, y_true)


class DiceCoefficientLoss(ERCLoss):
    """
    Dice Coefficient Loss function for ERC models.

    Reference:
    >>> Peiqing Lv, Jinke Wang, Xiangyang Zhang, Chunlei Ji, Lubiao Zhou, and Haiying Wang. 2022.
    ... An improved residual U-Net with morphological-based loss function for automatic liver
    ... segmentation in computed tomography. Math. Biosci. Eng. 19, 2 (January 2022), 1426–1447.
    """

    def __init__(self, config: ERCConfig):
        super().__init__(config=config)
        self.epsilon = config.classifier_epsilon

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate and return the loss given the predicted and true labels
        using the Dice Loss function, which is defined as mean of the Dice
        coefficient across all classes, and derived from the equation:
        >>> 1 - (2 * TP) / (2 * TP + FP + FN)

        :param y_pred: Predicted labels, already converted to a batch of softmax probability distributions
        :param y_true: True labels
        :return: Loss value
        """
        # Compute TP, FP, and FN for each class
        assert y_true is not None
        TP = (y_pred * y_true).sum(dim=0)
        FP = (y_pred * (1 - y_true)).sum(dim=0)
        FN = ((1 - y_pred) * y_true).sum(dim=0)

        # Compute Dice coefficient for each class
        dice_class = (2 * TP) / (2 * TP + FP + FN + self.epsilon)

        # Average Dice coefficient across all classes and compute the loss
        return 1 - dice_class.mean()


class FocalMutiClassLogLoss(ERCLoss):
    """
    Focal Multi-class Log Loss function for ERC models.

    Reference:
    >>> Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2020.
    ... Focal Loss for Dense Object Detection. IEEE Transactions on Pattern Analysis
    ... and Machine Intelligence 42, 2 (2020), 318–327. DOI:https://doi.org/10.1109/TPAMI.2018.2858826
    """

    def __init__(self, config: ERCConfig):
        super().__init__(config=config)
        alpha = config.losses_focal_alpha
        gamma = config.losses_focal_gamma
        reduction = config.losses_focal_reduction
        epsilon = config.classifier_epsilon
        if not (isinstance(alpha, list) and len(alpha) == len(config.classifier_classes)):
            raise ValueError("alpha must be a list of length equal to the number of classes")
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma if gamma else 2.0
        self.reduction = reduction if reduction else "mean"
        self.epsilon = epsilon

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates and returns the loss given the predicted and true labels
        using the Focal Cross Entropy Loss function, which is defined as:
        >>> -alpha * (1 - p)^gamma * log(p)

        :param y_pred: Predicted labels, already converted to a batch of softmax probability distributions
        :param y_true: True labels
        :return: Loss value
        """
        # ensure forware pre-reqs
        assert y_true is not None
        self.alpha = self.alpha if self.alpha.device == y_pred.device else self.alpha.to(y_pred.device)
        # compute the loss
        probs = torch.sum(y_pred * y_true, dim=1)
        safe_probs = torch.clamp(probs, min=self.epsilon, max=1.0 - self.epsilon)
        alpha = self.alpha[y_true.argmax(dim=1)]
        focal_weights = alpha * (1.0 - safe_probs).pow(self.gamma)
        loss = focal_weights * -torch.log(safe_probs)
        return loss.mean() if self.reduction == "mean" else loss.sum()


class TripletLoss(torch.nn.Module):
    """
    Triplet Loss function for ERC models, implemented using the equation designed
    by the SimCSE model, introduced by Gao et al. (2021), given by the equation:
    >>> -log(
    ...    exp(sim(a,p)/temperature) /
    ...    sum((exp(sim(a,p)/temperature) + exp(sim(a,n)/temperature))
    ... )

    However, the SimCSE paper fine-tunes a hyperparameter for the temperature,
    which would require significant experimentation to find the optimal value,
    however much more expensive due to the multi-modal nature of the model.

    For that reason, instead of cosine similarity with a temperature, we use
    the dot product, which will be made equivalent to cosine similarity by
    changing the embeddings produced by the model prior to logits to be l2-norm.

    As a consequence, the temperature hyperparameter is no longer required and
    the triplet loss equation implemented here is:
    >>> -log(
    ...    exp(dot(a,p.T)) /
    ...    sum((exp(dot(a,p.T)) + exp(dot(a,n.T)))
    ... )

    Reference:
    >>> Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive
    ... Learning of Sentence Embeddings. In EMNLP 2021 - 2021 Conference on Empirical
    ... Methods in Natural Language Processing, Proceedings (EMNLP 2021 - 2021
    ... Conference on Empirical Methods in Natural Language Processing, Proceedings),
    ... Association for Computational Linguistics (ACL), 6894–6910.
    """

    def __init__(self, config: ERCConfig):
        super().__init__()

    def forward(
        self,
        y_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculate and return the loss given the predicted and true labels
        using the Triplet Loss function defined in the SimCSE paper by the following equation:
        >>> -log(
        ...    exp(dot(a,p.T)) /
        ...    sum((exp(dot(a,p.T)) + exp(dot(a,n.T)))
        ... )

        :param y_embeddings: Normalised l2-norm embeddings produced before logits are calculated.
        :return: Loss value
        """
        anchor, positive, negative = y_embeddings
        assert anchor.shape == positive.shape == negative.shape
        anchor_positive_dot = torch.dot(anchor, positive.T)
        anchor_negative_dot = torch.dot(anchor, negative.T)
        numerator = torch.exp(anchor_positive_dot)
        denominator = torch.exp(anchor_positive_dot) + torch.exp(anchor_negative_dot)
        loss = -torch.log(numerator / denominator)
        return loss.mean()
