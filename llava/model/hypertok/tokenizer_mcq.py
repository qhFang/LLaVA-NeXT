import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .unitok_mcq import GeGluMlp, PlainAttention


class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = PlainAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(
            in_features=out_dim,
            hidden_features=hidden_dim
        )

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x




class MCQ(nn.Module):
    """
    Multi-Codebook Quantizer (RVQ-style), drop-in replacement for FSQ.

    Input : z  -> (B, N, input_dim)
    Output: out -> (B, N, input_dim)
    Also returns commitment loss.
    """

    def __init__(
        self,
        num_codes: int,
        input_dim: int,
        code_dim: int,
        num_codebooks: int = 1,
        commitment_weight: float = 1.0,
        projection_has_bias: bool = True,
    ):
        super().__init__()


        self.num_codebooks = num_codebooks
        self.commitment_weight = commitment_weight

        self.dim = code_dim * num_codebooks
        self.codebook_dim = code_dim 
        assert input_dim % num_codebooks == 0, \
            "input_dim must be divisible by num_codebooks"

        # optional projection (kept for interface compatibility)
        self.project_in = nn.Linear(
            input_dim, self.dim, bias=projection_has_bias
        )
        self.project_out = nn.Linear(
            self.dim, input_dim, bias=projection_has_bias
        )

        # codebooks: one embedding table per book
        self.codebooks = nn.ParameterList([
            nn.Parameter(
                torch.randn(num_codes, self.codebook_dim) * 0.02
            )
            for i in range(num_codebooks)
        ])

    def quantize(self, z):
        """
        z: (B, N, C, D)
        returns:
          z_q : quantized vectors, same shape
          commit_loss
        """
        z_q = []
        commit_loss = 0.0

        for i in range(self.num_codebooks):
            zi = z[:, :, i]                         # (B, N, D_cb)
            cb = self.codebooks[i]                 # (K, D_cb)

            # squared L2 distance
            dist = (
                zi.unsqueeze(-2) - cb.unsqueeze(0).unsqueeze(0)
            ).pow(2).sum(-1)                        # (B, N, K)

            idx = dist.argmin(dim=-1)               # (B, N)
            zqi = cb[idx]                           # (B, N, D_cb)

            # commitment loss
            commit_loss = commit_loss + F.mse_loss(
                zi.detach(), zqi, reduction="mean"
            )

            # straight-through estimator
            zqi = zi + (zqi - zi).detach()
            z_q.append(zqi)

        z_q = torch.stack(z_q, dim=2)               # (B, N, C, D)
        return z_q, self.commitment_weight * commit_loss

    def forward(self, z):
        """
        z: (B, N, input_dim)
        """

        z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        z_q, commit_loss = self.quantize(z)

        z_q = rearrange(z_q, "b n c d -> b n (c d)")
        out = self.project_out(z_q)

        return out, commit_loss

