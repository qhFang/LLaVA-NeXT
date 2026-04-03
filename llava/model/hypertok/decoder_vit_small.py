from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .timm_vitamin import GeGluMlp, ViTaminDecoder
from timm.models.vision_transformer import Attention, LayerScale
from timm.models.vision_transformer import Block as TimmBlock
from timm.layers import DropPath
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List


class SimpleMultiheadAttentionBasis(nn.Module):
    """
    自定义 Multi-Head Attention，每个 q/k/v/out 投影都是 BasisLinear。
    所以 hypernet 会给 q/k/v/out 各自一组 coeffs。
    """
    def __init__(self, dim: int, num_heads: int, num_basis: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = BasisLinear(dim, dim, num_basis)
        self.k_proj = BasisLinear(dim, dim, num_basis)
        self.v_proj = BasisLinear(dim, dim, num_basis)
        self.out_proj = BasisLinear(dim, dim, num_basis)

    def forward(
        self,
        x: torch.Tensor,
        coeffs: dict,
    ) -> torch.Tensor:
        """
        x: (B, N, D)
        coeffs: dict with keys ["q", "k", "v", "out"], each (B, K)
        """
        B, N, D = x.shape

        q = self.q_proj(x, coeffs["q"])  # (B, N, D)
        k = self.k_proj(x, coeffs["k"])
        v = self.v_proj(x, coeffs["v"])

        # reshape to multi-head
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, Hd)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,N,N)
        attn = attn_scores.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B,H,N,Hd)

        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        out = self.out_proj(out, coeffs["out"])  # (B, N, D)
        return out


class TransformerBlock(nn.Module):
    """
    标准 transformer block:
      - self.attn: q, k, v, out  各一层
      - MLP: fc1, fc2
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        """
        B, N, D = x.shape

        # Self-Attention
        h = self.norm1(x)
        h = h.permute(1, 0, 2)  # (N, B, D)
        attn_out, _ = self.attn(h, h, h)
        attn_out = attn_out.permute(1, 0, 2)  # (B, N, D)
        x = x + attn_out

        # MLP
        h = self.norm2(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        x = x + h
        return x


class TransformerBlockHyper(nn.Module):
    """
    单个 transformer block，所有线性层均为 BasisLinear：
      - self.attn: q, k, v, out  各一层
      - MLP: fc1, fc2
    超网络对该 block 提供 6 组 coeffs：
      index: 0:q, 1:k, 2:v, 3:out, 4:fc1, 5:fc2
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        num_basis: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_basis = num_basis

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimpleMultiheadAttentionBasis(dim, num_heads, num_basis)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = BasisLinear(dim, hidden, num_basis)
        self.fc2 = BasisLinear(hidden, dim, num_basis)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, coeffs_block: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        coeffs_block: (B, 6, K)  六个线性层的 coeffs
        index 映射:
          0 -> q, 1 -> k, 2 -> v, 3 -> out, 4 -> fc1, 5 -> fc2
        """
        B, N, D = x.shape
        assert coeffs_block.shape[0] == B
        assert coeffs_block.shape[1] == 6

        # 拆分各层 coeffs
        coeff_q = coeffs_block[:, 0, :]  # (B, K)
        coeff_k = coeffs_block[:, 1, :]
        coeff_v = coeffs_block[:, 2, :]
        coeff_out = coeffs_block[:, 3, :]
        coeff_fc1 = coeffs_block[:, 4, :]
        coeff_fc2 = coeffs_block[:, 5, :]

        # Self-Attention
        h = self.norm1(x)
        attn_out = self.attn(
            h,
            {
                "q": coeff_q,
                "k": coeff_k,
                "v": coeff_v,
                "out": coeff_out,
            },
        )
        x = x + attn_out

        # MLP
        h = self.norm2(x)
        h = self.fc1(h, coeff_fc1)
        h = self.act(h)
        h = self.fc2(h, coeff_fc2)
        x = x + h
        return x

class SimpleCrossAttention(nn.Module):
    """
    Cross-attention:
      Q from x
      K,V from mem
    output: x + CrossAttn(x, mem)
    """
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, Nq, D)
        mem: (B, Nm, D)
        """
        B, Nq, D = x.shape
        _, Nm, _ = mem.shape

        q = self.q(x).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nq, Hd)
        k = self.k(mem).view(B, Nm, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, Nm, Hd)
        v = self.v(mem).view(B, Nm, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, Nm, Hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nm)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, Nq, Hd)
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)  # (B, Nq, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossInjectBlock(nn.Module):
    """
    x <- x + CrossAttn(LN(x), mem)
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = SimpleCrossAttention(dim, num_heads)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        return x + self.cross(self.norm_q(x), self.norm_kv(mem))


class HyperFiLMBlock(nn.Module):
    """Transformer block with pre-normalization with Hypernet and FiLM after pre-norm."""

    def __init__(
            self,
            dim: int = 1024,
            num_heads: int = 16,
            mlp_ratio: float = 2.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = GeGluMlp,
            query_num: int = 16,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.hyper_embedding = nn.Parameter(torch.randn(1, query_num, dim) * 0.02)
        self.hyper_cross1 = CrossInjectBlock(dim, num_heads)
        self.hyper_cross2 = CrossInjectBlock(dim, num_heads)
        self.hyper_proj = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.hyper_proj.weight)
        nn.init.zeros_(self.hyper_proj.bias)

        
    def forward(self, x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        h = self.hyper_embedding.expand(x.size(0), -1, -1).contiguous()
        h = self.hyper_cross1(h, token)
        h = self.hyper_cross2(h, token)
        film_params = self.hyper_proj(h).mean(1)   # [B, 2C]

        film_weights = 1. + 0.1 * torch.tanh(film_params[:, :x.size(-1)]).unsqueeze(1)  # [B, 1, C]
        film_bias = 0.1 * torch.tanh(film_params[:, x.size(-1):]).unsqueeze(1)     # [B, 1, C]

        x_norm = self.norm1(x)
        x_film = x_norm * film_weights + film_bias

        x = x + self.drop_path1(self.ls1(self.attn(x_film)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def act_layer_from_mlp(mlp: nn.Module) -> Type[nn.Module]:
    if hasattr(mlp, "act") and isinstance(mlp.act, nn.Module):
        return mlp.act.__class__
    for m in mlp.modules():
        if isinstance(m, (nn.GELU, nn.ReLU, nn.SiLU, nn.LeakyReLU)):
            return m.__class__
    return nn.GELU


def convert_timm_block_to_hyper_film_block(
    old_block: nn.Module,
    hyper_block_cls,
    depth: int = 0,
    dim: int = 1024,
    num_heads: int = 16,
    mlp_ratio: float = 2.,
    mlp_layer: nn.Module = GeGluMlp,
    query_num: int = 16,

):
    dim = old_block.norm1.normalized_shape[0]

    if not hasattr(old_block.attn, "num_heads"):
        raise AttributeError("Cannot infer num_heads from old_block.attn.num_heads")
    num_heads = old_block.attn.num_heads


    new_block = hyper_block_cls(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        mlp_layer=mlp_layer,
        query_num=query_num,
    )

    # copy pretrained submodules
    new_block.norm1.load_state_dict(old_block.norm1.state_dict())
    new_block.attn.load_state_dict(old_block.attn.state_dict())
    new_block.norm2.load_state_dict(old_block.norm2.state_dict())
    new_block.mlp.load_state_dict(old_block.mlp.state_dict())

    if hasattr(old_block, "ls1") and hasattr(new_block, "ls1"):
        if type(old_block.ls1) is type(new_block.ls1):
            new_block.ls1.load_state_dict(old_block.ls1.state_dict())

    if hasattr(old_block, "ls2") and hasattr(new_block, "ls2"):
        if type(old_block.ls2) is type(new_block.ls2):
            new_block.ls2.load_state_dict(old_block.ls2.state_dict())

    return new_block


def replace_blocks_with_hyperfilm(
    module: nn.Module,
    block_cls,
    hyper_block_cls,
    film_layer_num: int = 5,
    verbose: bool = True,
    **kwargs,
):
    for name, child in list(module.named_children()):
        
        if isinstance(child, block_cls):
            #filter the first film_layer_num blocks to replace
            if int(name) >= film_layer_num:
                continue

            new_block = convert_timm_block_to_hyper_film_block(
                old_block=child,
                hyper_block_cls=hyper_block_cls,
                depth=0,
                **kwargs,
            )
            setattr(module, name, new_block)
            if verbose:
                print(f"Replaced {name}: {block_cls.__name__} -> {hyper_block_cls.__name__}")
        else:
            replace_blocks_with_hyperfilm(
                module=child,
                block_cls=block_cls,
                hyper_block_cls=hyper_block_cls,
                verbose=verbose,
                **kwargs,
            )




class DecoderClass(nn.Module):
    def __init__(self):
        super().__init__()

class VitaminDecoder(DecoderClass):
    def __init__(
        self,
        model,
        num_query=0,
        img_size=256,
        drop_path=0.,
        depths=(4, 2),
        grad_ckpt=False,
        in_dim=768,
        hidden_dim=768,
    ):
        super().__init__()

        self.vis_decoder = ViTaminDecoder(
            model,
            num_query=0,
            img_size=256,
            drop_path=0.1,
            grad_ckpt=True,
        )
        self.fc_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.projection = nn.Linear(in_dim, hidden_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0)))
        #self.logit_bias = nn.Parameter(torch.zeros([]))
        self.logit_bias = torch.tensor(0.0)

    def get_last_param(self):
        return self.vis_decoder.get_last_param()


    def forward(self, x, task="vis"):
        if task == "vis":
            return self.vis_decoder(x)
        elif task == "sem":
            clip_visual = x.mean(dim=1)
            clip_visual = self.projection(self.fc_norm(clip_visual))
            clip_visual = F.normalize(clip_visual, dim=-1)

            return clip_visual, self.logit_scale.exp(), self.logit_bias.to(self.logit_scale.device)



class VitaminDecoderHyperFilm(DecoderClass):
    def __init__(
        self,
        model,
        num_query=0,
        img_size=256,
        drop_path=0.,
        depths=(4, 2),
        grad_ckpt=False,
        in_dim=768,
        hidden_dim=768,
        query_num=16,
        film_layer_num=5,
        num_heads=16,
    ):
        super().__init__()

        self.vis_decoder = ViTaminDecoder(
            model,
            num_query=0,
            img_size=256,
            drop_path=0.1,
            grad_ckpt=True,
            film_layer_num=film_layer_num,
        )

        if model == "vitamin_large":
            replace_blocks_with_hyperfilm(
                self.vis_decoder.blocks,
                block_cls=TimmBlock,
                hyper_block_cls=HyperFiLMBlock,
                film_layer_num=film_layer_num,
                **{
                    "dim": 1024,
                    "num_heads": 16,
                    "mlp_ratio": 2.,
                    "mlp_layer": GeGluMlp,
                    "query_num": query_num,
                }
            )
        else:
            raise ValueError(f"Unsupported model: {model}")

        self.clip_hyper_embedding = nn.Parameter(torch.randn(1, query_num, in_dim) * 0.02)
        self.clip_hyper_cross1 = CrossInjectBlock(in_dim, num_heads)
        self.clip_hyper_cross2 = CrossInjectBlock(in_dim, num_heads)
        self.clip_hyper_proj = nn.Linear(in_dim, 2 * in_dim)
        nn.init.zeros_(self.clip_hyper_proj.weight)
        nn.init.zeros_(self.clip_hyper_proj.bias)

        self.fc_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.projection = nn.Linear(in_dim, hidden_dim)



        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0)))
        #self.logit_bias = nn.Parameter(torch.zeros([]))
        self.logit_bias = torch.tensor(0.0)

    def get_last_param(self):
        return self.vis_decoder.get_last_param()


    def forward(self, x, task="vis"):
        if task == "vis":
            return self.vis_decoder(x)
        elif task == "sem":
            h = self.clip_hyper_cross1(self.clip_hyper_embedding.expand(x.size(0), -1, -1).contiguous(), x)
            h = self.clip_hyper_cross2(h, x)
            clip_film_params = self.clip_hyper_proj(h.mean(1))   # [B, C]
            clip_film_weight, clip_film_bias = clip_film_params.chunk(2, dim=-1)  # [B, C], [B, C]

            clip_visual = x.mean(dim=1) * (1. + 0.1 * torch.tanh(clip_film_weight)) + 0.1 * torch.tanh(clip_film_bias)
            clip_visual = self.projection(self.fc_norm(clip_visual))
            clip_visual = F.normalize(clip_visual, dim=-1)

            return clip_visual, self.logit_scale.exp(), self.logit_bias.to(self.logit_scale.device)



if __name__ == "__main__":
    model = VitaminDecoderHyperFilm(
        model="vitamin_large",        
        num_query=0,
        img_size=256,
        drop_path=0.,
        depths=(4, 2),
        grad_ckpt=False,
        in_dim=1025,
        hidden_dim=1024,
        query_num=16,
        film_layer_num=5,
    )
    print(model.vis_decoder.blocks)
