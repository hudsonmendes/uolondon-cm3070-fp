# Python Built-in Modules
from dataclasses import dataclass


@dataclass(frozen=True)
class RetinaFaceArgs:
    """
    Replace the command line arguments from the original code with a dataclass.
    """

    network: str = "resnet50"
    cpu: bool = False
    confidence_threshold: float = 0.02
    top_k: int = 10
    nms_threshold: float = 0.4
    keep_top_k: int = 750
    save_image: bool = True
    vis_thres: float = 0.6
