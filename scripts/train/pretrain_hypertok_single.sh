#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

LLM_VERSION="/data/share/Qwen3.5-9B/"
LLM_VERSION_CLEAN=LLM_VERSION
VISION_MODEL_VERSION="hypertok"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

HYPERTOK_CKPT="/data/250010203/aisports/Codes/HyperTok/outputs/alldata_128fsq_timmvitamin_deltafilm/ckpt_epoch1.pt"
HYPERTOK_CONFIG="/data/250010203/aisports/Codes/HyperTok/configs/default.yaml"
HYPERTOK_FEATURE_SOURCE="decoder_sem"  # quant or decoder_sem
HYPERTOK_QUANTIZER="fsq"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /data/share/250010203/data/recap558k/data \
    --image_folder /blip_558k/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_pretrained ${HYPERTOK_CKPT} \
    --hypertok_config ${HYPERTOK_CONFIG} \
    --hypertok_feature_source ${HYPERTOK_FEATURE_SOURCE} \
    --hypertok_quantizer ${HYPERTOK_QUANTIZER} \
    --use_hypertok_tfm True \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /data/250010203/aisports/Codes/LLaVA-NeXT/checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa \
    --qwen3_5_vl_weights True
