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
        # testing a Gaussian Fourier feature mappings with scale 10
        # from Tancik's paper on Fourier Features
        torch.manual_seed(0)
        B_gauss = torch.randn((self.num_octaves * 2, samples.shape[-1])) * 10
        x_proj = (2.0 * torch.pi * samples) @ B_gauss.T
        embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

        return embedding

    def d_out(self, dimensionality: int):
        return self.num_octaves * 2
