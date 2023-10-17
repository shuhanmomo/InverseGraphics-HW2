import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves: int, mode: str = "basic"):
        super().__init__()
        self.num_octaves = num_octaves
        if mode not in ("basic", "gaussian"):
            raise ValueError("The 'mode' parameter must be 'basic' or 'gaussian'")
        self.mode = mode

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        """Separately encode each channel using a positional encoding. The lowest
        frequency should be 2 * torch.pi, and each frequency thereafter should be
        double the previous frequency. For each frequency, you should encode the input
        signal using both sine and cosine.
        """
        # gaussian mode tests a Gaussian Fourier feature mappings with scale 10
        # from Tancik's paper on Fourier Features
        torch.manual_seed(0)
        coord_dim = samples.shape[-1]
        if self.mode == "gaussian":
            mapping_size = 256
            B = (
                torch.randn((self.num_octaves * mapping_size * coord_dim, coord_dim))
                * 10
            )

        elif self.mode == "basic":
            freqs = (
                2.0 ** torch.arange(0, self.num_octaves).float().to(samples.device)
                * 2
                * np.pi
            )
            B = (
                freqs[:, None]
                .repeat(1, coord_dim)
                .repeat(coord_dim, 1)
                .view(-1, coord_dim)
            )
        B = B.to(samples.device)
        x_proj = (2.0 * torch.pi * samples) @ B.T
        embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

        return embedding

    def d_out(self, dimensionality: int):
        if self.mode == "basic":
            d_out = self.num_octaves * dimensionality * 2
        if self.mode == "gaussian":
            d_out = 2 * self.num_octaves * 256 * dimensionality
        return d_out
