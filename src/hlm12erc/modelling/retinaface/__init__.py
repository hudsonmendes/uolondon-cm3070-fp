# Python Built-in Modules
from typing import Callable

# Third-Party Libraries
import torch

# Local Folders
from .config import cfg_re50
from .detection import detect, load_model
from .model import RetinaFace


def create_face_detector(filepath_pretrained: str) -> Callable[[str], dict]:
    device = _get_device()
    net = _load_from_pretrained(filepath_pretrained, device)
    return lambda x: detect(net, x, device)


def _get_device() -> torch.device:
    device = torch.cpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    return device


def _load_from_pretrained(filepath_pretrained: str, device: torch.device):
    net = RetinaFace(cfg=cfg_re50, phase="test")
    net = load_model(net, filepath_pretrained, load_to_cpu=device == torch.cpu)
    net.eval()
    return net
