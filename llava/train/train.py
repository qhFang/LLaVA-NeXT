# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import os
import copy
import inspect
from contextlib import contextmanager
import io
import bisect
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np

import time
import random
import yaml
import math
import re
import torch

import transformers
import tokenizers
import deepspeed

from transformers import AutoConfig
from transformers.modeling_utils import load_state_dict
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
from torch.utils.data import Dataset
from torchvision import transforms
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)

    hypertok_config: Optional[str] = field(default=None, metadata={"help": "Path to HyperTok YAML config (optional)."})
    hypertok_encoder: Optional[str] = field(default="vitamin_large")
    hypertok_image_size: Optional[int] = field(default=256)
    hypertok_embed_dim: Optional[int] = field(default=None)
    hypertok_quantizer: Optional[str] = field(default="fsq", metadata={"help": "fsq|mcq|unitokmcq|unitokr|none"})
    hypertok_levels: Optional[str] = field(default=None, metadata={"help": "Comma-separated levels for FSQ, e.g. 7 or 7,7,7"})
    hypertok_num_codebooks: Optional[int] = field(default=128)
    hypertok_num_codes: Optional[int] = field(default=32768)
    hypertok_code_dim: Optional[int] = field(default=64)
    hypertok_feature_source: Optional[str] = field(default="quant", metadata={"help": "quant|decoder_sem"})
    hypertok_decoder_variant: Optional[str] = field(default="ours", metadata={"help": "ours|nohyper"})
    hypertok_decoder_query_num: Optional[int] = field(default=16)
    hypertok_decoder_film_layer_num: Optional[int] = field(default=5)
    hypertok_decoder_hidden_dim: Optional[int] = field(default=768)
    hypertok_decoder_num_heads: Optional[int] = field(default=16)
    qwen3_5_vl_weights: Optional[bool] = field(
        default=None,
        metadata={
            "help": "If True, load Qwen3.5 VL checkpoints by remapping model.language_model.* -> model.*. "
            "If False, expect text-only checkpoints. If None, auto-detect from config."
        },
    )



@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)
    parquet_image_column: Optional[str] = field(default="image")
    parquet_conversation_column: Optional[str] = field(default="conversations")
    parquet_id_column: Optional[str] = field(default="id")
    use_hypertok_tfm: bool = field(default=False)


class HypertokImageProcessor:
    def __init__(self, size: int):
        self.size = size
        self.crop_size = {"height": size, "width": size}
        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]
        self._tfm = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )

    def preprocess(self, image, return_tensors="pt"):
        if isinstance(image, list):
            tensor_list = [self._tfm(img) for img in image]
            return {"pixel_values": torch.stack(tensor_list, dim=0)}
        tensor = self._tfm(image)
        return {"pixel_values": tensor.unsqueeze(0)}


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    group_by_length: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    trainer_mode: str = field(default="regular", metadata={"help": "Mode of training (`regular`, `zo`)."})
    zo_eps: float = field(default=1e-3, metadata={"help": "MeZO hyperparameter epsilon."})
    zo_num_directions: int = field(default=1, metadata={"help": "Number of directions for MeZO."})
    include_tokens_per_second: bool = field(default=False)
    past_index: int = field(default=-1)
    dispatch_batches: Optional[bool] = field(default=None)
    split_batches: Optional[bool] = field(default=None)
    deepspeed_plugin: Optional[object] = field(default=None)

    def to_dict(self):
        data = super().to_dict()
        if "deepspeed_plugin" in data:
            data["deepspeed_plugin"] = None
        return data


# @dataclass
# class EvaluationArguments:
#     eval_num_processes: int = field(default=1)
#     task_names: str = field(default=None)
#     model: str = field(default="llava")
#     model_args: Optional[str] = field(default=None)
#     num_fewshot: Optional[int] = field(default=None)
#     batch_size: int = field(default=1)
#     device: Optional[str] = field(default=None)
#     limit: Optional[int] = field(default=None)
#     check_integrity: Optional[bool] = field(default=False)
#     show_task_to_terminal: Optional[bool] = field(default=False)
#     log_samples: Optional[bool] = field(default=True)
#     gen_kwargs: Optional[str] = field(default="")
#     log_samples_suffix: Optional[str] = field(default="")
#     output_path: Optional[str] = field(default="./logs/")


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
    elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    if check_only_save_mm_adapter_tunnable:
        # Only save Adapter
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    alt_image_token = "<images>"
    for source in sources:
        for sentence in source:
            if alt_image_token in sentence["value"]:
                sentence["value"] = sentence["value"].replace(alt_image_token, DEFAULT_IMAGE_TOKEN)
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    additional_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
    if not additional_ids:
        additional_tokens = getattr(tokenizer, "additional_special_tokens", []) or []
        additional_ids = tokenizer.convert_tokens_to_ids(additional_tokens)

    if len(additional_ids) >= 2:
        im_start, im_end = additional_ids[:2]
    else:
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_start is None or im_end is None:
            raise ValueError("Missing Qwen special tokens <|im_start|> / <|im_end|> in tokenizer.")
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    def _token_to_ids(token: str):
        vocab = tokenizer.get_vocab()
        if token in vocab:
            return [tokenizer.convert_tokens_to_ids(token)]
        return tokenizer(token).input_ids

    def _coerce_token_ids(seq):
        flat = []
        for item in seq:
            if isinstance(item, int):
                flat.append(item)
                continue
            if isinstance(item, str):
                flat.extend(_token_to_ids(item))
                continue
            if isinstance(item, (list, tuple, np.ndarray)):
                flat.extend(_coerce_token_ids(list(item)))
                continue
            flat.extend(tokenizer(str(item)).input_ids)
        return flat

    def apply_chat_template_ids(messages):
        # Ensure we always get token ids even if apply_chat_template defaults to string.
        try:
            output = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        except TypeError:
            output = tokenizer.apply_chat_template(messages)
        if hasattr(output, "input_ids"):
            output = output.input_ids
        elif isinstance(output, dict) and "input_ids" in output:
            output = output["input_ids"]
        if isinstance(output, str):
            return tokenizer(output).input_ids
        return _coerce_token_ids(output)

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += apply_chat_template_ids([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            encode_id = apply_chat_template_ids(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        input_id = _coerce_token_ids(input_id)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    def _token_to_ids(token: str):
        vocab = tokenizer.get_vocab()
        if token in vocab:
            return [tokenizer.convert_tokens_to_ids(token)]
        return tokenizer(token).input_ids

    def _coerce_token_ids(seq):
        flat = []
        for item in seq:
            if isinstance(item, int):
                flat.append(item)
                continue
            if isinstance(item, str):
                flat.extend(_token_to_ids(item))
                continue
            if isinstance(item, (list, tuple, np.ndarray)):
                flat.extend(_coerce_token_ids(list(item)))
                continue
            flat.extend(tokenizer(str(item)).input_ids)
        return flat

    def apply_chat_template_ids(messages):
        # Ensure we always get token ids even if apply_chat_template defaults to string.
        try:
            output = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        except TypeError:
            output = tokenizer.apply_chat_template(messages)
        if hasattr(output, "input_ids"):
            output = output.input_ids
        elif isinstance(output, dict) and "input_ids" in output:
            output = output["input_ids"]
        if isinstance(output, str):
            return tokenizer(output).input_ids
        return _coerce_token_ids(output)

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += apply_chat_template_ids([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = apply_chat_template_ids(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        input_id = _coerce_token_ids(input_id)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = []
        self._parquet_total_rows = 0
        self._parquet_cum_rows = []
        self._parquet_row_groups = {}
        self._parquet_files = {}
        self._parquet_warned = False
        self._parquet_length_cache = {}
        self._parquet_modality_cache = {}
        self.data_args = data_args

        def append_parquet_rows(rows):
            image_col = data_args.parquet_image_column
            conv_col = data_args.parquet_conversation_column
            id_col = data_args.parquet_id_column
            for row in rows:
                conversations = row.get(conv_col)
                if conversations is None and "conversations" in row:
                    conversations = row.get("conversations")
                if conversations is None and "conversation" in row:
                    conversations = row.get("conversation")
                if isinstance(conversations, np.ndarray):
                    conversations = conversations.tolist()
                if conversations is None:
                    raise ValueError(f"Missing conversations column in parquet row. Expected '{conv_col}' or 'conversations'.")
                sample = {"conversations": conversations}
                if image_col in row:
                    image_value = row.get(image_col)
                    if isinstance(image_value, dict) and "bytes" in image_value:
                        image_value = image_value.get("bytes")
                    sample["image"] = image_value
                if id_col in row and row.get(id_col) is not None:
                    sample["id"] = row.get(id_col)
                self.list_data_dict.append(sample)

        # Handle multiple JSON files specified in the data_path
        if os.path.isdir(data_path):
            parquet_files = sorted(pathlib.Path(data_path).glob("*.parquet"))
            if not parquet_files:
                raise ValueError(f"No .parquet files found in directory: {data_path}")
            try:
                import pyarrow.parquet as pq
            except Exception as exn:
                raise ImportError("pyarrow is required to read parquet datasets. Please install pyarrow.") from exn
            data_args.dataset_paths = [str(p) for p in parquet_files]
            for parquet_path in parquet_files:
                rank0_print(f"Loading {parquet_path}")
                parquet_file = pq.ParquetFile(parquet_path)
                num_rows = parquet_file.metadata.num_rows
                self._parquet_total_rows += num_rows
                self._parquet_cum_rows.append((self._parquet_total_rows, str(parquet_path)))
                row_group_sizes = [
                    parquet_file.metadata.row_group(i).num_rows
                    for i in range(parquet_file.metadata.num_row_groups)
                ]
                row_group_cum = []
                total = 0
                for size in row_group_sizes:
                    total += size
                    row_group_cum.append(total)
                self._parquet_row_groups[str(parquet_path)] = row_group_cum
        elif data_path.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
            except Exception as exn:
                raise ImportError("pyarrow is required to read parquet datasets. Please install pyarrow.") from exn
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            parquet_file = pq.ParquetFile(data_path)
            num_rows = parquet_file.metadata.num_rows
            self._parquet_total_rows += num_rows
            self._parquet_cum_rows.append((self._parquet_total_rows, str(data_path)))
            row_group_sizes = [
                parquet_file.metadata.row_group(i).num_rows
                for i in range(parquet_file.metadata.num_row_groups)
            ]
            row_group_cum = []
            total = 0
            for size in row_group_sizes:
                total += size
                row_group_cum.append(total)
            self._parquet_row_groups[str(data_path)] = row_group_cum
        elif "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.list_data_dict) + self._parquet_total_rows

    def _get_parquet_row(self, parquet_path: str, row_idx: int) -> Dict:
        try:
            import pyarrow.parquet as pq
        except Exception as exn:
            raise ImportError("pyarrow is required to read parquet datasets. Please install pyarrow.") from exn

        if parquet_path not in self._parquet_files:
            self._parquet_files[parquet_path] = pq.ParquetFile(parquet_path)
        parquet_file = self._parquet_files[parquet_path]
        row_group_cum = self._parquet_row_groups[parquet_path]
        row_group_idx = bisect.bisect_right(row_group_cum, row_idx)
        row_group_start = 0 if row_group_idx == 0 else row_group_cum[row_group_idx - 1]
        row_in_group = row_idx - row_group_start
        table = parquet_file.read_row_group(row_group_idx)
        row = table.slice(row_in_group, 1).to_pylist()[0]
        return row

    def _parquet_row_to_sample(self, row: Dict, parquet_path: Optional[str] = None, row_idx: Optional[int] = None) -> Dict:
        image_col = self.data_args.parquet_image_column
        conv_col = self.data_args.parquet_conversation_column
        id_col = self.data_args.parquet_id_column
        conversations = row.get(conv_col)
        if conversations is None and "conversations" in row:
            conversations = row.get("conversations")
        if conversations is None and "conversation" in row:
            conversations = row.get("conversation")
        if isinstance(conversations, np.ndarray):
            conversations = conversations.tolist()
        if isinstance(conversations, str):
            try:
                conversations = json.loads(conversations)
            except Exception as exn:
                raise ValueError(
                    f"Parquet row has string conversations that is not JSON. "
                    f"path={parquet_path}, row={row_idx}, type={type(conversations)}"
                ) from exn
        if not isinstance(conversations, list):
            raise ValueError(
                f"Parquet row conversations is not a list. path={parquet_path}, row={row_idx}, "
                f"type={type(conversations)}"
            )
        if conversations is None:
            raise ValueError(f"Missing conversations column in parquet row. Expected '{conv_col}' or 'conversations'.")
        cleaned_convs = []
        for conv in conversations:
            if isinstance(conv, np.ndarray):
                conv = conv.tolist()
            if isinstance(conv, str):
                conv = {"from": "human", "value": conv}
            if not isinstance(conv, dict):
                raise ValueError(
                    f"Conversation item is not dict. path={parquet_path}, row={row_idx}, type={type(conv)}"
                )
            value = conv.get("value", conv.get("content", ""))
            if not isinstance(value, str):
                value = str(value)
            conv["value"] = value
            cleaned_convs.append(conv)
        sample = {"conversations": cleaned_convs}
        if parquet_path is not None:
            sample["_parquet_path"] = parquet_path
        if row_idx is not None:
            sample["_parquet_row"] = row_idx
        if image_col in row:
            image_value = row.get(image_col)
            if isinstance(image_value, dict) and "bytes" in image_value:
                image_value = image_value.get("bytes")
            sample["image"] = image_value
        if id_col in row and row.get(id_col) is not None:
            sample["id"] = row.get(id_col)
        return sample

    def _get_length_from_sample(self, sample: Dict) -> int:
        conversations = sample.get("conversations", [])
        total = 0
        for conv in conversations:
            text = conv.get("value")
            if text is None:
                text = conv.get("content", "")
            total += len(str(text).split())
        if "image" in sample:
            total += 128
        return total

    def _get_modality_length_from_sample(self, sample: Dict) -> int:
        conversations = sample.get("conversations", [])
        total = 0
        for conv in conversations:
            text = conv.get("value")
            if text is None:
                text = conv.get("content", "")
            total += len(str(text).split())
        if total <= 0:
            raise ValueError(f"Conversation length is 0 for {sample}")
        if "image" in sample or "video" in sample or self.data_args.early_mix_text:
            return total
        return -total

    def _get_parquet_sample(self, parquet_idx: int) -> Dict:
        if not self._parquet_cum_rows:
            raise IndexError("Parquet index out of range and no parquet files were loaded.")
        parquet_file_idx = bisect.bisect_right(
            [end for end, _ in self._parquet_cum_rows], parquet_idx
        )
        parquet_start = 0 if parquet_file_idx == 0 else self._parquet_cum_rows[parquet_file_idx - 1][0]
        parquet_path = self._parquet_cum_rows[parquet_file_idx][1]
        row_idx = parquet_idx - parquet_start
        row = self._get_parquet_row(parquet_path, row_idx)
        try:
            return self._parquet_row_to_sample(row, parquet_path=parquet_path, row_idx=row_idx)
        except Exception as exn:
            raise ValueError(f"Invalid parquet row. path={parquet_path}, row={row_idx}") from exn

    @property
    def lengths(self):
        dataset = self

        class _LazyLengths:
            def __len__(self):
                return len(dataset)

            def __getitem__(self, idx):
                if idx < len(dataset.list_data_dict):
                    return dataset._get_length_from_sample(dataset.list_data_dict[idx])
                parquet_idx = idx - len(dataset.list_data_dict)
                cached = dataset._parquet_length_cache.get(parquet_idx)
                if cached is not None:
                    return cached
                sample = dataset._get_parquet_sample(parquet_idx)
                length = dataset._get_length_from_sample(sample)
                dataset._parquet_length_cache[parquet_idx] = length
                return length

            def __iter__(self):
                for i in range(len(dataset)):
                    yield self[i]

        return _LazyLengths()

    @property
    def modality_lengths(self):
        dataset = self

        class _LazyModalityLengths:
            def __len__(self):
                return len(dataset)

            def __getitem__(self, idx):
                if idx < len(dataset.list_data_dict):
                    return dataset._get_modality_length_from_sample(dataset.list_data_dict[idx])
                parquet_idx = idx - len(dataset.list_data_dict)
                cached = dataset._parquet_modality_cache.get(parquet_idx)
                if cached is not None:
                    return cached
                sample = dataset._get_parquet_sample(parquet_idx)
                length = dataset._get_modality_length_from_sample(sample)
                dataset._parquet_modality_cache[parquet_idx] = length
                return length

            def __iter__(self):
                for i in range(len(dataset)):
                    yield self[i]

        return _LazyModalityLengths()

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            if isinstance(image_file, (bytes, bytearray, memoryview)):
                image = Image.open(io.BytesIO(image_file)).convert("RGB")
            else:
                image_path = os.path.join(image_folder, image_file) if image_folder is not None else image_file
                image = Image.open(image_path).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        if i < len(self.list_data_dict):
            sources = self.list_data_dict[i]
        else:
            parquet_idx = i - len(self.list_data_dict)
            sources = self._get_parquet_sample(parquet_idx)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]

        image = None
        has_image = False

        if "image" in sample:
            image_file = sample.get("image", None)
            # Some datasets keep the 'image' key but store None for missing images.
            # Treat these as text-only samples to avoid crashing in process_image().
            if image_file is None or image_file == "" or (isinstance(image_file, list) and len(image_file) == 0):
                sources = copy.deepcopy([e["conversations"] for e in sources])
            else:
                has_image = True
                if type(image_file) is list:
                    # Filter out None entries if present.
                    image_file = [f for f in image_file if f is not None and f != ""]
                    if len(image_file) == 0:
                        has_image = False
                        sources = copy.deepcopy([e["conversations"] for e in sources])
                    else:
                        image = [self.process_image(f) for f in image_file]
                        # Handling multi images: overwrite to process with simple pad
                        if len(image_file) > 1:
                            image = [self.process_image(f, "pad") for f in image_file]
                            image = [[im[0], im[1], "image"] for im in image]
                else:
                    image = [self.process_image(image_file)]

                if has_image:
                    sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        elif "video" in sample:
            video_file = sample["video"]
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
            suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))

            try:
                if "shareVideoGPTV" in video_file:
                    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                    if self.data_args.force_sample:
                        num_frames_to_sample = self.data_args.frames_upbound
                    else:
                        num_frames_to_sample = 10

                    avg_fps = 2

                    total_frames = len(frame_files)
                    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)


                    frame_time = [i/2 for i in sampled_indices]
                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                    video_time = total_frames / avg_fps

                    # Read and store the sampled frames
                    video = []
                    for idx in sampled_indices:
                        frame_path = frame_files[idx]
                        try:
                            with Image.open(frame_path) as img:
                                frame = img.convert("RGB")
                                video.append(frame)
                        except IOError:
                            print(f"Failed to read frame at path: {frame_path}")
                else:
                    video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, video[0].size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print(sources)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = has_image or ("video" in sample)
        try:
            data_dict = preprocess(sources, self.tokenizer, has_image=has_image)
        except Exception as exn:
            parquet_path = sample.get("_parquet_path")
            parquet_row = sample.get("_parquet_row")
            sample_id = sample.get("id", i)
            if parquet_path is not None or parquet_row is not None:
                raise ValueError(
                    f"Preprocess failed. id={sample_id}, path={parquet_path}, row={parquet_row}"
                ) from exn
            raise

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if has_image and image is not None:
            data_dict["image"] = image
        elif "video" in sample:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = sample.get("id", i)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def _resolve_qwen3_5_checkpoint_files(model_name_or_path: str, cache_dir: Optional[str] = None) -> list[str]:
    index_names = [SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME]
    for index_name in index_names:
        try:
            index_file = cached_file(model_name_or_path, index_name, cache_dir=cache_dir)
            shard_files, _ = get_checkpoint_shard_files(model_name_or_path, index_file)
            return shard_files
        except Exception:
            pass

    weight_names = [SAFE_WEIGHTS_NAME, WEIGHTS_NAME]
    for weight_name in weight_names:
        try:
            weight_file = cached_file(model_name_or_path, weight_name, cache_dir=cache_dir)
            return [weight_file]
        except Exception:
            pass

    raise FileNotFoundError(f"Could not find weights in {model_name_or_path}")


def _load_qwen3_5_state_dict(model_name_or_path: str, cache_dir: Optional[str] = None) -> Dict[str, torch.Tensor]:
    state_dict = {}
    for shard_file in _resolve_qwen3_5_checkpoint_files(model_name_or_path, cache_dir=cache_dir):
        shard_state = load_state_dict(shard_file)
        state_dict.update(shard_state)
    return state_dict


def _remap_qwen3_5_vl_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("model.language_model."):
            new_key = "model." + key[len("model.language_model.") :]
            remapped[new_key] = value
            continue
        if key.startswith("model.visual") or key.startswith("model.vision_tower") or key.startswith("vision_tower"):
            continue
        remapped[key] = value
    return remapped


def _log_loading_info(loading_info: Optional[Dict], model_tag: str) -> None:
    if not loading_info:
        rank0_print(f"Loading {model_tag} without loading_info.")
        return
    missing = loading_info.get("missing_keys") or []
    unexpected = loading_info.get("unexpected_keys") or []
    mismatched = loading_info.get("mismatched_keys") or []
    error_msgs = loading_info.get("error_msgs") or []

    rank0_print(
        "Loading info for %s: missing=%d unexpected=%d mismatched=%d errors=%d"
        % (model_tag, len(missing), len(unexpected), len(mismatched), len(error_msgs))
    )
    if missing:
        rank0_print("Missing keys (first 20): %s" % missing[:20])
    if unexpected:
        rank0_print("Unexpected keys (first 20): %s" % unexpected[:20])
    if mismatched:
        rank0_print("Mismatched keys (first 20): %s" % mismatched[:20])
    if error_msgs:
        rank0_print("Loading errors (first 5): %s" % error_msgs[:5])


def _from_pretrained_with_info(model_class, *args, **kwargs):
    loading_info = None
    try:
        result = model_class.from_pretrained(*args, output_loading_info=True, **kwargs)
    except TypeError:
        result = model_class.from_pretrained(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        model, loading_info = result
    else:
        model = result
    _log_loading_info(loading_info, getattr(model_class, "__name__", str(model_class)))
    return model


@contextmanager
def _guard_zero3_empty_embedding_init():
    orig_init_weights = transformers.modeling_utils.PreTrainedModel._init_weights

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Embedding):
            weight = module.weight
            if weight is None or weight.numel() == 0:
                return
            padding_idx = getattr(module, "padding_idx", None)
            if padding_idx is not None and padding_idx >= weight.shape[0]:
                return
        return orig_init_weights(self, module)

    transformers.modeling_utils.PreTrainedModel._init_weights = _init_weights
    try:
        yield
    finally:
        transformers.modeling_utils.PreTrainedModel._init_weights = orig_init_weights


@contextmanager
def _disable_deepspeed_zero3_init():
    try:
        from deepspeed import zero
    except Exception:
        yield
        return
    with zero.Init(enabled=False):
        yield


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    if model_args.model_class_name is not None:
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        rank0_print(f"Using model class {model_class} from {model_args.model_class_name}")
        model = _from_pretrained_with_info(
            model_class,
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    elif model_args.vision_tower is not None:
        if "mixtral" in model_args.model_name_or_path.lower():
            model = _from_pretrained_with_info(
                LlavaMixtralForCausalLM,
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            model = _from_pretrained_with_info(
                LlavaMistralForCausalLM,
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            model = _from_pretrained_with_info(
                LlavaLlamaForCausalLM,
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif "qwen3" in model_args.model_name_or_path.lower() or "qwen3_5" in model_args.model_name_or_path.lower() or "qwen3.5" in model_args.model_name_or_path.lower():
            cfg_full = AutoConfig.from_pretrained(model_args.model_name_or_path)
            cfg_full_dict = cfg_full.to_dict()
            cfg_pretrained = getattr(cfg_full, "text_config", None)
            if cfg_pretrained is None:
                text_config_dict = cfg_full_dict.get("text_config", cfg_full_dict)
                cfg_pretrained = Qwen3_5TextConfig.from_dict(text_config_dict)

            vocab_size = (
                (cfg_full_dict.get("text_config") or {}).get("vocab_size")
                or cfg_full_dict.get("vocab_size")
                or getattr(cfg_pretrained, "vocab_size", 0)
            )
            if not vocab_size:
                raise ValueError("Qwen3.5 config is missing vocab_size.")
            cfg_pretrained.vocab_size = int(vocab_size)
            if getattr(cfg_pretrained, "pad_token_id", None) is None:
                cfg_pretrained.pad_token_id = getattr(cfg_pretrained, "eos_token_id", None) or 0
            if cfg_pretrained.pad_token_id is not None and cfg_pretrained.pad_token_id >= cfg_pretrained.vocab_size:
                fallback_pad = getattr(cfg_pretrained, "eos_token_id", 0)
                cfg_pretrained.pad_token_id = fallback_pad if fallback_pad < cfg_pretrained.vocab_size else 0
            cfg_pretrained.model_type = "llava_qwen3_5"
            customized_kwargs["config"] = cfg_pretrained
            vl_weights = model_args.qwen3_5_vl_weights
            if vl_weights is None:
                vl_weights = bool(cfg_full_dict.get("vision_config") or cfg_full_dict.get("image_token_id"))
            if vl_weights:
                state_dict = _load_qwen3_5_state_dict(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
                remapped_state_dict = _remap_qwen3_5_vl_state_dict(state_dict)
                pretrained_kwargs = dict(customized_kwargs)
                pretrained_kwargs.pop("config", None)
                # NOTE: use HF low_cpu_mem_usage path to avoid CPU OOM at init.
                with _guard_zero3_empty_embedding_init():
                    model = _from_pretrained_with_info(
                        LlavaQwen3_5ForCausalLM,
                        None,
                        state_dict=remapped_state_dict,
                        cache_dir=training_args.cache_dir,
                        attn_implementation=training_args.attn_implementation,
                        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                        low_cpu_mem_usage=True,
                        config=cfg_pretrained,
                        **pretrained_kwargs,
                    )
            else:
                model = _from_pretrained_with_info(
                    LlavaQwen3_5ForCausalLM,
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
        elif "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                model = _from_pretrained_with_info(
                    LlavaQwenMoeForCausalLM,
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                model = _from_pretrained_with_info(
                    LlavaQwenForCausalLM,
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
        elif "gemma" in model_args.model_name_or_path.lower():
            model = _from_pretrained_with_info(
                LlavaGemmaForCausalLM,
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        else:
            raise ValueError(f"Unknown model class {model_args}")
    else:
        model = _from_pretrained_with_info(
            transformers.LlamaForCausalLM,
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    return model


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_3_5"]


    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        if data_args.use_hypertok_tfm and model_args.vision_tower == "hypertok":
            data_args.image_processor = HypertokImageProcessor(size=model_args.hypertok_image_size)

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                if type(data_args.image_processor.size) is int:
                    patch_size = data_args.image_processor.size
                else:
                    try:
                        patch_size = data_args.image_processor.size[0]
                    except Exception as e:
                        patch_size = data_args.image_processor.size["shortest_edge"]

                assert patch_size in [224, 256, 336, 384, 448, 512], "patch_size should be in [224, 256, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.add_time_instruction = data_args.add_time_instruction
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride

        ### Deciding train which part of the model
        if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
            if training_args.freeze_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)
            else:
                vision_tower.requires_grad_(False)

        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            # Set the entire model to not require gradients by default
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            model.get_model().vision_resampler.requires_grad_(False)
            # Parse the mm_tunable_parts to decide which parts to unfreeze
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_resampler" in tunable_parts:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                        param.requires_grad_(True)

        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer_kwargs = dict(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer_sig = inspect.signature(transformers.Trainer.__init__)
    if "tokenizer" not in trainer_sig.parameters:
        trainer_kwargs.pop("tokenizer", None)
        if "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
    trainer = LLaVATrainer(**trainer_kwargs)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
