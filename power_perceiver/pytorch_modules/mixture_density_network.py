from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class MixtureDensityNetwork(nn.Module):
    def __init__(self, in_features: int, num_gaussians: int = 2):
        super().__init__()
        self.pi = nn.Linear(in_features=in_features, out_features=num_gaussians)
        self.mu = nn.Linear(in_features=in_features, out_features=num_gaussians)
        self.sigma = nn.Linear(in_features=in_features, out_features=num_gaussians)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pi = self.pi(x)
        pi = F.softmax(pi, dim=-1)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = torch.exp(sigma)
        return torch.concat((pi, mu, sigma), dim=-1)


def get_distribution(
    network_output: torch.Tensor,
    example_i: Optional[int] = None,
    num_gaussians: int = 2,
) -> torch.distributions.MixtureSameFamily:

    pi = network_output[..., :num_gaussians]
    mu = network_output[..., num_gaussians : num_gaussians * 2]
    sigma = network_output[..., num_gaussians * 2 :]

    if example_i is not None:
        pi = pi[example_i]
        mu = mu[example_i]
        sigma = sigma[example_i]

    mixture_distribution = torch.distributions.Categorical(probs=pi)
    component_distribution = torch.distributions.Normal(loc=mu, scale=sigma)
    gaussian_mixture_model = torch.distributions.MixtureSameFamily(
        mixture_distribution, component_distribution
    )
    return gaussian_mixture_model
