import logging
from functools import partial

import torch
from fastai.layers import Mish
from fastai.vision.learner import create_unet_model
from fastai.vision.models.xresnet import xse_resnext50
from torch import nn

_log = logging.getLogger(__name__)


class XResUNet(nn.Module):
    """Predict future satellite images using a U-Net.

    This model was developed by the Illinois team in ClimateHack.AI in early 2022.
    This model won the ClimateHack.AI competition :). See
    https://github.com/jmather625/climatehack
    """

    def __init__(self, arch=None, **kwargs):
        """Create a DynamicUnet.

        See this page for a description of the kwargs:
        https://fastai1.fast.ai/vision.models.unet.html#DynamicUnet
        """
        _log.info(f"kwargs for XResUNet = {kwargs}")
        super().__init__()
        if arch is None:
            arch = partial(xse_resnext50, act_cls=Mish, sa=True)
        self.model = create_unet_model(arch=arch, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
