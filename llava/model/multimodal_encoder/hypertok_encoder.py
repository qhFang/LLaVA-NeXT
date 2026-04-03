import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor
import yaml
import timm
from timm.data import resolve_model_data_config

from llava.utils import rank0_print
from llava.model.hypertok import (
    GeGluMlp,
    VitaminDecoder,
    VitaminDecoderHyperFilm,
    FSQ,
    MCQ,
    VectorQuantizerM,
    VectorQuantizerR,
)


class HyperTokVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.args = args
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.hypertok_config = self._load_hypertok_config(getattr(args, "hypertok_config", None))

        self.encoder_name = getattr(args, "hypertok_encoder", None) or self._cfg_get("encoder", "variant", default="vitamin_large")
        self.image_size = int(getattr(args, "hypertok_image_size", None) or self._cfg_get("image_size", None, root_key="data", default=256))
        self.quantizer_type = (getattr(args, "hypertok_quantizer", None) or self._cfg_get("tokenizer", "name", default="fsq")).lower()
        self.feature_source = getattr(args, "hypertok_feature_source", "quant")

        self._embed_dim = int(getattr(args, "hypertok_embed_dim", None) or self._cfg_get("encoder", "embed_dim", default=1024))

        self._levels = self._parse_levels(getattr(args, "hypertok_levels", None) or self._cfg_get("tokenizer", "levels", default=[7]))
        self._num_codebooks = int(getattr(args, "hypertok_num_codebooks", None) or self._cfg_get("tokenizer", "num_codebooks", default=128))
        self._num_codes = int(getattr(args, "hypertok_num_codes", None) or self._cfg_get("tokenizer", "num_codes", default=32768))
        self._code_dim = int(getattr(args, "hypertok_code_dim", None) or self._cfg_get("tokenizer", "code_dim", default=64))

        self.decoder_variant = getattr(args, "hypertok_decoder_variant", "ours")
        self.decoder_query_num = int(getattr(args, "hypertok_decoder_query_num", None) or self._cfg_get("decoder", "query_num", default=16))
        self.decoder_film_layer_num = int(getattr(args, "hypertok_decoder_film_layer_num", None) or self._cfg_get("decoder", "film_layer_num", default=5))
        self.decoder_hidden_dim = int(getattr(args, "hypertok_decoder_hidden_dim", 768))
        self.decoder_num_heads = int(getattr(args, "hypertok_decoder_num_heads", 16))

        self._use_decoder = self.feature_source == "decoder_sem"

        if not delay_load:
            rank0_print(f"Loading HyperTok vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            rank0_print("The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print("The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self._config = SimpleNamespace(
                image_size=self.image_size,
                patch_size=1,
                hidden_size=self.hidden_size,
            )

    def _cfg_get(self, section: str, key: Optional[str], root_key: str = "model", default: Any = None) -> Any:
        if not self.hypertok_config:
            return default
        block = self.hypertok_config.get(root_key, {})
        if section in block:
            value = block[section]
            if key is None:
                return value
            return value.get(key, default)
        return default

    def _load_hypertok_config(self, path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        if not os.path.exists(path):
            rank0_print(f"HyperTok config not found: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _parse_levels(self, levels_value):
        if levels_value is None:
            return [7]
        if isinstance(levels_value, (list, tuple)):
            return list(levels_value)
        if isinstance(levels_value, str):
            return [int(x) for x in levels_value.split(",") if x]
        return [int(levels_value)]

    def _load_weights(self, checkpoint_path: str):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return
        rank0_print(f"Loading HyperTok weights from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
        encoder_state = self._extract_prefixed_state(state, "encoder")
        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)
        quant_state = self._extract_prefixed_state(state, "fsq")
        if quant_state and self.quantizer is not None:
            self.quantizer.load_state_dict(quant_state, strict=False)
        decoder_state = self._extract_prefixed_state(state, "decoder")
        if decoder_state and self.decoder is not None:
            self.decoder.load_state_dict(decoder_state, strict=False)

    @staticmethod
    def _extract_prefixed_state(state: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        if prefix in state and isinstance(state[prefix], dict):
            return state[prefix]
        out = {}
        prefix_dot = prefix + "."
        for key, value in state.items():
            if key.startswith(prefix_dot):
                out[key[len(prefix_dot):]] = value
        return out

    def _build_encoder(self):
        encoder = timm.create_model(
            self.encoder_name,
            patch_size=1,
            fc_norm=False,
            drop_rate=0.0,
            num_classes=0,
            global_pool="",
            pos_embed="none",
            class_token=False,
            mlp_layer=GeGluMlp,
            reg_tokens=0,
            img_size=self.image_size,
            drop_path_rate=0.1,
        )
        if hasattr(encoder, "set_grad_checkpointing"):
            encoder.set_grad_checkpointing(True)
        return encoder

    def _build_quantizer(self, embed_dim: int):
        if self.quantizer_type == "fsq":
            return FSQ(
                levels=self._levels,
                input_dim=embed_dim,
                num_codebooks=self._num_codebooks,
            )
        if self.quantizer_type == "mcq":
            return MCQ(
                input_dim=embed_dim,
                num_codes=self._num_codes,
                code_dim=self._code_dim,
                num_codebooks=self._num_codebooks,
            )
        if self.quantizer_type == "unitokmcq":
            return VectorQuantizerM(
                input_dim=embed_dim,
                vocab_size=self._num_codes,
                vocab_width=self._code_dim,
                num_codebooks=self._num_codebooks,
            )
        if self.quantizer_type == "unitokr":
            return VectorQuantizerR(
                input_dim=embed_dim,
                vocab_size=self._num_codes,
                vocab_width=self._code_dim,
                num_codebooks=self._num_codebooks,
            )
        if self.quantizer_type in ["none", ""]:
            return None
        raise ValueError(f"Unknown HyperTok quantizer: {self.quantizer_type}")

    def _build_decoder(self, embed_dim: int):
        if not self._use_decoder:
            return None
        if self.decoder_variant == "ours":
            return VitaminDecoderHyperFilm(
                self.encoder_name,
                num_query=0,
                img_size=self.image_size,
                drop_path=0.1,
                grad_ckpt=True,
                in_dim=embed_dim,
                hidden_dim=self.decoder_hidden_dim,
                query_num=self.decoder_query_num,
                film_layer_num=self.decoder_film_layer_num,
                num_heads=self.decoder_num_heads,
            )
        if self.decoder_variant == "nohyper":
            return VitaminDecoder(
                self.encoder_name,
                num_query=0,
                img_size=self.image_size,
                drop_path=0.1,
                grad_ckpt=True,
                in_dim=embed_dim,
                hidden_dim=self.decoder_hidden_dim,
            )
        raise ValueError(f"Unknown HyperTok decoder variant: {self.decoder_variant}")

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return

        self.encoder = self._build_encoder()
        self.encoder.requires_grad_(False)

        embed_dim = getattr(self.encoder, "embed_dim", None) or getattr(self.encoder, "num_features", None)
        if embed_dim is None:
            raise ValueError("Unable to infer HyperTok encoder embedding dimension.")
        self._embed_dim = int(embed_dim)

        self.quantizer = self._build_quantizer(self._embed_dim)
        if self.quantizer is not None:
            self.quantizer.requires_grad_(False)

        self.decoder = self._build_decoder(self._embed_dim)
        if self.decoder is not None:
            self.decoder.requires_grad_(False)

        data_cfg = resolve_model_data_config(self.encoder)
        mean = data_cfg.get("mean", (0.5, 0.5, 0.5))
        std = data_cfg.get("std", (0.5, 0.5, 0.5))
        self.image_processor = CLIPImageProcessor(
            size={"shortest_edge": self.image_size},
            crop_size={"height": self.image_size, "width": self.image_size},
            image_mean=list(mean),
            image_std=list(std),
        )
        rank0_print(f"Loaded HyperTok image processor: {self.image_processor}")

        self._config = SimpleNamespace(
            image_size=self.image_size,
            patch_size=self._get_patch_size(),
            hidden_size=self.hidden_size,
        )

        ckpt_path = getattr(self.args, "vision_tower_pretrained", None)
        self._load_weights(ckpt_path)

        self.is_loaded = True

    def _get_patch_size(self):
        patch_embed = getattr(self.encoder, "patch_embed", None)
        if patch_embed is None:
            return 1
        patch_size = getattr(patch_embed, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            return patch_size[0]
        if isinstance(patch_size, int):
            return patch_size
        return 1

    def _quantize(self, z):
        if self.quantizer is None:
            return z
        out = self.quantizer(z)
        if isinstance(out, tuple):
            return out[0]
        return out

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self._forward_once(image.unsqueeze(0))
                image_features.append(image_feature)
            return image_features
        return self._forward_once(images)

    def _forward_once(self, images):
        images = images.to(device=self.device, dtype=self.dtype)
        z = self.encoder(images)
        if z.dim() == 2:
            z = z.unsqueeze(1)
        z_q = self._quantize(z)

        if self._use_decoder:
            sem_out, _, _ = self.decoder(z_q, task="sem")
            if sem_out.dim() == 2:
                sem_out = sem_out.unsqueeze(1)
            return sem_out.to(images.dtype)

        return z_q.to(images.dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @property
    def config(self):
        return self._config

    @property
    def hidden_size(self):
        if self._use_decoder:
            return self.decoder_hidden_dim
        return self._embed_dim

    @property
    def num_patches_per_side(self):
        if self._use_decoder:
            return 1
        patch_embed = getattr(self.encoder, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "grid_size"):
            return patch_embed.grid_size[0]
        return self.image_size // self._get_patch_size()

    @property
    def num_patches(self):
        if self._use_decoder:
            return 1
        patch_embed = getattr(self.encoder, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "num_patches"):
            return patch_embed.num_patches
        return (self.image_size // self._get_patch_size()) ** 2

    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, value):
        self._image_size = int(value)
