import datetime

import matplotlib
import matplotlib.dates as mdates
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
        # mu = F.relu(mu)  # Guarantee that PV can't go negative!
        sigma = self.sigma(x)
        sigma = torch.exp(sigma)
        return torch.concat((pi, mu, sigma), dim=-1)


def get_pi_mu_sigma(
    network_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert network_output.shape[-1] % 3 == 0
    num_gaussians = int(network_output.shape[-1] / 3)
    pi = network_output[..., :num_gaussians]
    mu = network_output[..., num_gaussians : num_gaussians * 2]
    sigma = network_output[..., num_gaussians * 2 :]
    return pi, mu, sigma


def get_distribution(network_output: torch.Tensor) -> torch.distributions.MixtureSameFamily:
    pi, mu, sigma = get_pi_mu_sigma(network_output)

    mixture_distribution = torch.distributions.Categorical(probs=pi)
    component_distribution = torch.distributions.Normal(loc=mu, scale=sigma)
    gaussian_mixture_model = torch.distributions.MixtureSameFamily(
        mixture_distribution, component_distribution
    )
    return gaussian_mixture_model


def plot_probs(
    ax: matplotlib.axes.Axes,
    network_output: torch.Tensor,
    left: datetime.datetime,
    right: datetime.datetime,
) -> matplotlib.axes.Axes:
    """Plot distribution over time.

    Args:
        network_output: Shape: (time, mixture_density_network_features)
        left, right: Pass in the output of `mdates.date2num` for the left and
            right time boundaries.
    """
    left = mdates.date2num(left)
    right = mdates.date2num(right)

    sweep_n_steps = 100
    sweep_start = 1
    sweep_stop = 0

    n_timesteps = network_output.shape[0]

    # Define a 'sweep' matrix which we pass into log_prob to get probabilities
    # for a range of values at each timestep. Those values range from sweep_start to sweep_stop.
    sweep = torch.linspace(
        start=sweep_start,
        end=sweep_stop,
        steps=sweep_n_steps,
        dtype=torch.float32,
        device=network_output.device,
    )
    sweep = sweep.unsqueeze(-1).expand(sweep_n_steps, n_timesteps)

    # Get probabilities.
    distribution = get_distribution(network_output)
    log_probs = distribution.log_prob(sweep)
    probs = torch.exp(log_probs).detach().cpu().numpy()

    # Normalise so every column has a max of 1 otherwise blurry distributions are very hard to see!
    probs /= probs.max(axis=0)

    # Plot!
    extent = (left, right, sweep_stop, sweep_start)  # left, right, bottom, top
    ax.imshow(probs, aspect="auto", interpolation="none", extent=extent, cmap="Greys")

    return ax
