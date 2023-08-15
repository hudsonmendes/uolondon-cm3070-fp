# My Packages and Modules
from hlm12erc.modelling import ERCConfig


class ERCConfigFormatter:
    """Represents a ERCConfig as a string."""

    def __init__(self, config: ERCConfig) -> None:
        """
        Constructs a new instance of ERCConfigFormatter.

        :param config: ERCConfig to be represented as a string
        """
        self.config = config

    def represent(self) -> str:
        """
        Represents the ERCConfig as a string, with sufficient information
        to make one model configuration distinguishable from another.

        :return: String representation of the ERCConfig
        """
        return "-".join(["hlm12erc", self.config.classifier_name.lower()])
