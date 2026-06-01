import argparse
import copy
import json
import os

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Simple single-image chat demo for LLaVA-NeXT checkpoints.")
    parser.add_argument("--ckpt", required=True, help="Path to the checkpoint or model directory.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--prompt",
        default="Please describe this image in detail.",
        help="User prompt for the model.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model family name for load_pretrained_model. If omitted, infer from checkpoint config.",
    )
    parser.add_argument(
        "--conv-template",
        default=None,
        help="Optional conversation template. If omitted, infer from checkpoint config.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Inference dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation passed to load_pretrained_model, e.g. sdpa or flash_attention_2.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for generation when sampling is enabled.")
    parser.add_argument("--device", default="cuda", help="Torch device for inputs. Usually cuda.")
    return parser.parse_args()


def infer_runtime_defaults(ckpt_path: str, model_name: str | None, conv_template: str | None):
    resolved_model_name = model_name
    resolved_conv_template = conv_template

    config_path = os.path.join(ckpt_path, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(x).lower() for x in config.get("architectures", [])]

    if resolved_model_name is None:
        if "qwen3_5" in model_type or "llavaqwen3_5forcausallm" in architectures:
            resolved_model_name = "llava_qwen3_5"
        elif "qwen" in model_type or any("qwen" in arch for arch in architectures):
            resolved_model_name = "llava_qwen"
        elif "mistral" in model_type or any("mistral" in arch for arch in architectures):
            resolved_model_name = "llava_mistral"
        elif "gemma" in model_type or any("gemma" in arch for arch in architectures):
            resolved_model_name = "llava_gemma"
        else:
            resolved_model_name = "llava_qwen"

    if resolved_conv_template is None:
        if "qwen" in resolved_model_name.lower():
            resolved_conv_template = "qwen_3_5"
        else:
            resolved_conv_template = "qwen_3_5"

    return resolved_model_name, resolved_conv_template


def main():
    args = parse_args()
    args.model_name, args.conv_template = infer_runtime_defaults(
        args.ckpt, args.model_name, args.conv_template
    )

    if args.conv_template not in conv_templates:
        raise ValueError(
            f"Unknown conv template: {args.conv_template}. "
            f"Available templates: {', '.join(sorted(conv_templates.keys()))}"
        )

    print(f"[demo] resolved model_name={args.model_name}, conv_template={args.conv_template}")

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.ckpt,
        model_base=None,
        model_name=args.model_name,
        device_map="auto",
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    model.eval()
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        vision_tower.to(device=args.device, dtype=torch_dtype)

    image = Image.open(args.image).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [x.to(device=args.device, dtype=torch_dtype) for x in image_tensor]
    else:
        image_tensor = image_tensor.to(device=args.device, dtype=torch_dtype)

    question = DEFAULT_IMAGE_TOKEN + "\n" + args.prompt
    conv = copy.deepcopy(conv_templates[args.conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)

    do_sample = args.temperature > 0
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            modalities=["image"],
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    generated_ids = output_ids[:, input_ids.shape[1] :]
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(output_text)


if __name__ == "__main__":
    main()
