from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from .field.field import Field
import torch
import numpy as np


class NeRF(nn.Module):
    cfg: DictConfig
    field: Field

    def __init__(self, cfg: DictConfig, field: Field) -> None:
        super().__init__()
        self.cfg = cfg
        self.field = field

    def forward(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
    ) -> Float[Tensor, "batch 3"]:
        """Render the rays using volumetric rendering. Use the following steps:

        1. Generate sample locations along the rays using self.generate_samples().
        2. Evaluate the neural field at the sample locations. The neural field's output
           has four channels: three for RGB color and one for volumetric density. Don't
           forget to map these channels to valid output ranges.
        3. Compute the alpha values for the evaluated volumetric densities using
           self.compute_alpha_values().
        4. Composite these alpha values together with the evaluated colors from.
        """

        raise NotImplementedError("This is your homework.")

    def generate_samples(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
        num_samples: int,
    ) -> tuple[
        Float[Tensor, "batch sample 3"],  # xyz sample locations
        Float[Tensor, "batch sample+1"],  # sample boundaries
    ]:
        """For each ray, equally divide the space between the specified near and far
        planes into num_samples segments. Return the segment boundaries (including the
        endpoints at the near and far planes). Also return sample locations, which fall
        at the midpoints of the segments.
        """
        num_rays = origins.shape[0]
        t_vals = torch.linspace(0.0, 1.0, steps=num_samples + 1)
        z_vals = (far - near) * t_vals + near
        z_vals = z_vals.expand([num_rays, num_samples + 1])
        mid_pts = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # stratified sampling
        np.random.seed(0)
        t_rand = torch.rand(*mid_pts.shape)
        upper = z_vals[..., 1:] - mid_pts
        lower = mid_pts - z_vals[..., :-1]
        z_vals_rand = mid_pts + upper * t_rand - lower * (1 - t_rand)
        pts = (
            origins[..., None, :] + directions[..., None, :] * z_vals_rand[..., :, None]
        )  # [N_rays, N_samples, 3]
        boundaries = z_vals

        return pts, boundaries

    def compute_alpha_values(
        self,
        sigma: Float[Tensor, "batch sample"],
        boundaries: Float[Tensor, "batch sample+1"],
    ) -> Float[Tensor, "batch sample"]:
        """Compute alpha values from volumetric densities (values of sigma) and segment
        boundaries.
        """
        seg_len = boundaries[..., 1:] - boundaries[..., :-1]
        alpha = 1 - torch.exp(-sigma * seg_len)
        return alpha

    def alpha_composite(
        self,
        alphas: Float[Tensor, "batch sample"],
        colors: Float[Tensor, "batch sample 3"],
    ) -> Float[Tensor, "batch 3"]:
        """Alpha-composite the supplied alpha values and colors. You may assume that the
        background is black.
        """
        T = torch.cumprod(1.0 - alphas + 1e-10, dim=-1)
        T = torch.cat(
            [torch.ones_like(T[..., :1], dtype=torch.float32), T[..., :-1]], dim=-1
        )
        weights = alphas * T
        rgb_map = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
        return rgb_map
