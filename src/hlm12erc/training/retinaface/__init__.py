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
    assert isinstance(device, torch.device) and device is not None
    assert isinstance(args, RetinaFaceArgs) and args is not None

    @torch.no_grad()
    def fn(filepath: str) -> list:
        return detect(model=net, device=device, args=args, filepath=str(filepath))

    return fn


def _load_from_pretrained(filepath_pretrained: str, device: torch.device):
    net = RetinaFace(cfg=cfg_re50, phase="test")
    net = load_model(net, filepath_pretrained, device=device)
    net.eval()
    return net
