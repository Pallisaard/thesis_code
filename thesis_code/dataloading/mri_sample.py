from typing import TypedDict

import torch


class MRISample(TypedDict):
    image: torch.Tensor
