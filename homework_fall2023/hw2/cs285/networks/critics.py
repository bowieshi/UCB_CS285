import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

        self.baseline_loss = nn.MSELoss()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        return self.network(obs)
        

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        # TODO: update the critic using the observations and q_values
        optimizer = self.optimizer
        optimizer.zero_grad()

        # Retrieve baseline output
        baseline_output = self(obs).squeeze()

        # Normalize q_values
        # q_values_mean = q_values.mean()
        # q_values_std = q_values.std()
        # q_values_normed = (q_values - q_values_mean) / (q_values_std + 1e-8)

        # Calculate loss
        # loss = self.baseline_loss(baseline_output, q_values_normed)
        loss = self.baseline_loss(baseline_output, q_values)

        # Update baseline parameters
        loss.backward()
        optimizer.step()

        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }