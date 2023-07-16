# Python Built-in Modules
from typing import Any, List, Union

# Third-Party Libraries
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder


class ERCLabelEncoder:
    """
    Wrapper to the Scikit Learn OneHotEncoder, simplifying the transformation
    of a single label into a MatrixLike object that can be more cleanly processed
    by th e fit/transformation methods.
    """

    def __init__(self, classes: List[str]) -> None:
        assert classes is not None
        assert isinstance(classes, list)
        assert len(classes) > 0
        self.one_hot = OneHotEncoder(sparse=False)
        self.one_hot.fit(np.array([[cls] for cls in classes]))

    @property
    def classes(self) -> List[str]:
        categories = self.one_hot.categories_[0]
        assert isinstance(categories, np.ndarray)
        return categories.tolist()

    def __call__(self, x: Union[str, List[str]]) -> torch.Tensor:
        """
        Callable interface to the encode method.
        """
        return self.encode(labels=x)

    def encode(self, labels: Union[str, List[str]]) -> torch.Tensor:
        """
        Encodes the given labels into a one-hot encoded matrix.

        :param labels: A single label or a list of labels to be encoded.
        :return: Either a single or a list of one-hot encoded labels
        """
        single = False
        if isinstance(labels, str):
            single = True
            labels = [labels]
        matrix = np.array([[label] for label in labels])
        matrix = self.one_hot.transform(matrix)
        assert isinstance(matrix, np.ndarray)
        matrix = matrix.tolist()
        out = matrix[0] if single else matrix
        return torch.tensor(out)
