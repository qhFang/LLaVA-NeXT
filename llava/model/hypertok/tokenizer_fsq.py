from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps, partial
from torch import tensor, int32
from einops import rearrange, pack, unpack

class FixedScaleQuantizer(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=4, n_levels=16, commitment_weight=1.0):
        super().__init__()
        self.project_in = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.project_out = nn.Linear(hidden_dim, in_dim) if in_dim != hidden_dim else nn.Identity()

        self.n_levels = n_levels
        self.commitment_weight = commitment_weight

        # per-channel scale (way more stable)
        self.log_scale = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, z):
        
        z_in = self.project_in(z)                   # (B,N,embed)

        scale = self.log_scale.exp().clamp(min=1e-4)
        z_scaled = z_in / scale

        # avoid collapse
        #z_clamped = z_scaled.clamp(-1, 1)

        levels = self.n_levels
        step = 2.0 / (levels - 1)

        codes = ((z_scaled + 1.0) / step).round().clamp(0, levels - 1)
        z_q_unit = codes * step - 1.0

        # commitment in embedding space
        commit_loss = F.mse_loss(z_scaled, z_q_unit.detach())


        z_q = z_q_unit * scale                      # unscale
        z_q = self.project_out(z_q)


        # straight-through
        z_q_st = z + (z_q - z).detach()
        
        return z_q_st, commit_loss, codes.long()


class FSQQuantizer(nn.Module):
    """Factorized Scalar Quantization with straight‑through gradient.

    Given feature x in R^{B×N×D}, we quantize each dimension to the closest level
    and use STE to pass gradients. Optionally, you can group dims and use separate
    level sets per group (omitted for brevity — keep one shared level set).
    """
    def __init__(self, levels: Tuple[float, ...] = (-2, -1, 0, 1, 2), in_dim: int = 768, hidden_dim: int = 16):
        super().__init__()
        lv = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels", lv)
        self.post_norm = nn.LayerNorm(hidden_dim)
        if in_dim != hidden_dim:
            self.proj_in = nn.Linear(in_dim, hidden_dim)
            self.proj_out = nn.Linear(hidden_dim, in_dim)
        else:
            self.proj_in = None
            self.proj_out = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # per‑batch/channel standardization for stability
        # x: [B,N,D]

        if self.proj_in is not None:
            x = self.proj_in(x)
        
        xn = self.post_norm(x)                # [B,N,D]
        # nearest level per element
        dist = (xn.unsqueeze(-1) - self.levels)**2  # [B,N,D,L]
        idx = dist.argmin(dim=-1)                   # [B,N,D]
        xq = self.levels[idx]
        # STE
        xq = xn + (xq - xn).detach()
        if self.proj_out is not None:
            xq = self.proj_out(xq)
        return xq, idx  # dequantized float, discrete indices

def round_ste(z):
    """ round with straight through gradients. """
    zhat = z.round()
    return z + (zhat - z).detach()

def floor_ste(z):
    """ floor with straight through gradients. """
    zhat = z.floor()
    return z + (zhat - z).detach()

def identity(t):
    return t





class FSQ(nn.Module):
    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        input_dim: int | None = None,
        num_codebooks = 1,
        scale: float | None = None,
        projection_has_bias = True,
        preserve_symmetry = False,
        noise_dropout = 0.,
        bound_hard_clamp = False # for residual fsq, if input is pre-softclamped to the right range
    ):
        super().__init__()

        if isinstance(levels, tuple):
            levels = list(levels)

        _levels = tensor(levels, dtype = int32)
        self.register_buffer('_levels', _levels, persistent = False)

        _basis = torch.cumprod(tensor([1] + levels[:-1]), dim = 0, dtype = int32)
        self.register_buffer('_basis', _basis, persistent = False)

        self.scale = scale

        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        


        self.dim = input_dim

        
        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.norm = nn.LayerNorm(effective_codebook_dim)

        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        # self.total_levels = self._levels * num_codebooks
        # cobook_usage = torch.zeros(self.total_levels)
        # self.register_buffer('cobook_usage', cobook_usage)





        # allow for a hard clamp

        self.bound_hard_clamp = bound_hard_clamp

    def bound(self, z, eps = 1e-3, hard_clamp = False):
        """ Bound `z`, an array of shape (..., d). """
        maybe_tanh = torch.tanh if not hard_clamp else partial(torch.clamp, min = -1., max = 1.)
        maybe_atanh = torch.atanh if not hard_clamp else identity

        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = maybe_atanh(offset / half_l)
        bounded_z = maybe_tanh(z + shift) * half_l - offset
        half_width = self._levels // 2
        round_z = round_ste(bounded_z)
        return round_z / half_width
        #return round_ste(bounded_z) / half_width, commit_loss

    # symmetry-preserving and noise-approximated quantization, section 3.2 in https://arxiv.org/abs/2411.19842
    
    def symmetry_preserving_bound(self, z, hard_clamp = False):
        """ QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1 """
        maybe_tanh = torch.tanh if not hard_clamp else partial(torch.clamp, min = -1., max = 1.)

        levels_minus_1 = (self._levels - 1)
        scale = 2. / levels_minus_1
        bracket = (levels_minus_1 * (maybe_tanh(z) + 1) / 2.) + 0.5
        bracket_backup = bracket
        bracket = floor_ste(bracket)
        return scale * bracket - 1.

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """

        shape, device, noise_dropout, preserve_symmetry = z.shape[0], z.device, self.noise_dropout, self.preserve_symmetry
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

    
        bounded_z = bound_fn(z, hard_clamp = self.bound_hard_clamp)

        # determine where to add a random offset elementwise
        # if using noise dropout

        if not self.training or noise_dropout == 0.:
            return bounded_z

        offset_mask = torch.bernoulli(torch.full_like(bounded_z, noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return bounded_z

    def _scale_and_shift(self, zhat_normalized):
        if self.preserve_symmetry:
            return (zhat_normalized + 1.) / (2. / (self._levels - 1))

        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        if self.preserve_symmetry:
            return zhat * (2. / (self._levels - 1)) - 1.

        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim = -1).round().to(int32)

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)


        codes = self._indices_to_codes(indices)
        codes = self.project_out(codes)


        return codes

    # def usage_update(self, indices):
    #     """ Update codebook usage statistics, for potential pruning or analysis. """
    #     indices = rearrange(indices, 'b n c d -> b n (c d)').detach()
    #     usage = torch.bincount(indices.flatten(), minlength = self.codebook_size * self.num_codebooks)

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)
        z = self.norm(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)
        
        orig_dtype = z.dtype

        codes = self.quantize(z)
        codes = rearrange(codes, 'b n c d -> b n (c d)')
        codes = codes.to(orig_dtype)

        # project out
        out = self.project_out(codes)
        
        return out, torch.tensor(0.0, device=out.device), torch.tensor(0.0, device=out.device), torch.tensor(0.0, device=out.device)


if __name__ == '__main__':
    # simple test
    quantizer = FSQ(levels = (8,6,5), input_dim = 8)
    x = torch.randn(2, 10, 8)
    xq = quantizer(x)
    print('Input:', x)
    print(xq)
