import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves: int):
        super().__init__()
        self.num_octaves = num_octaves

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        """Separately encode each channel using a positional encoding. The lowest
        frequency should be 2 * torch.pi, and each frequency thereafter should be
        double the previous frequency. For each frequency, you should encode the input
        signal using both sine and cosine.
        """

        coord_dim = samples.shape[1]
        identity = torch.eye(coord_dim)
        frequency = [2 * torch.pi * 2**i for i in range(self.num_octaves)]
        frequency = torch.cat([f * identity for f in frequency], dim=-1).to(
            samples.device
        )
        phase_angle = samples @ frequency

        embedding = torch.stack(
            [
                torch.sin(phase_angle),
                torch.cos(phase_angle),
            ],
            dim=-1,
        ).view(samples.shape[0], -1)

        return embedding

    def d_out(self, dimensionality: int):
        d_out = self.num_octaves * dimensionality * 2
        return d_out
