# Local Folders
from .meld_record import MeldRecord


class ERCDataCollator:
    """
    Collates the data from the ERC dataset into a format that can be used and
    batched by the model, with multiple records turned into matrices.
    """

    def __call__(self, record: MeldRecord) -> dict:
        raise NotImplementedError("Will implement it after Lara stops crying!!!")
