# Python Built-in Modules
from typing import Callable

# Third-Party Libraries
import torch

# Local Folders
from .args import RetinaFaceArgs
from .config import cfg_re50
from .detection import detect, load_model
from .model import RetinaFace

_cpu = torch.device("cpu")


def create_face_detector(
    filepath_pretrained: str,
    device: torch.device | None = None,
    args: RetinaFaceArgs | None = None,
) -> Callable[[str], list]:
    device = device or _cpu
    args = args or RetinaFaceArgs()
    net = _load_from_pretrained(filepath_pretrained, device)
    assert device is not None
    assert args is not None
    return lambda x: detect(model=net, device=device, args=args, filepath=str(x))


def _load_from_pretrained(filepath_pretrained: str, device: torch.device):
    net = RetinaFace(cfg=cfg_re50, phase="test")
    net = load_model(net, filepath_pretrained, load_to_cpu=device == _cpu)
    net.eval()
    return net
