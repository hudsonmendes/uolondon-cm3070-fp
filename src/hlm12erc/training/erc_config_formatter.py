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
        ff_layerspec = (
            "+".join([str(layer.out_features) for layer in self.config.feedforward_layers])
            if self.config.feedforward_layers
            else "default"
        )
        return "-".join(
            [
                "hlm12erc",
                self.config.modules_text_encoder.lower(),
                self.config.modules_visual_encoder.lower(),
                self.config.modules_audio_encoder.lower(),
                self.config.modules_fusion.lower(),
                f"t{self.config.text_in_features}x{self.config.text_out_features}",
                f"a{self.config.audio_in_features}x{self.config.audio_out_features}",
                f"ff{self.config.feedforward_out_features}",
                f"ffl{ff_layerspec}",
                f"{self.config.classifier_loss_fn}",
            ]
        )
