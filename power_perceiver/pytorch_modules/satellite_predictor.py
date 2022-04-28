from functools import partial

import torch
from fastai.layers import Mish
from fastai.vision.learner import create_unet_model
from fastai.vision.models.xresnet import xse_resnext50_deeper
from torch import nn


class XResUNet(nn.Module):
    """Predict future satellite images using a U-Net.

    This model was developed by the Illinois team in ClimateHack.AI in early 2022.
    This model won the ClimateHack.AI competition :). See
    https://github.com/jmather625/climatehack
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        forecast_steps: int,
        history_steps: int,
        pretrained: bool = False,
    ):
        super().__init__()
        arch = partial(xse_resnext50_deeper, act_cls=Mish, sa=True)
        self.model = create_unet_model(
            arch=arch,
            n_out=forecast_steps,
            img_size=input_size,
            pretrained=pretrained,
            n_in=history_steps,
            self_attention=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
