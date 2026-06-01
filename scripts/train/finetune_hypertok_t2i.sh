#!/bin/bash
set -euxo pipefail

export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_CROSS_NIC=0
export NCCL_ALGO=RING
export NCCL_SOCKET_IFNAME=eth0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_IB_AR_THRESHOLD=0
export OMP_NUM_THREADS=8

source /data/250010203/aisports/env/my_bashrc
source /data/250010203/aisports/env/.conda/etc/profile.d/conda.sh
cd /data/250010203/aisports/Codes/LLaVA-NeXT/
conda activate llm

LLM_VERSION="/data/share/Qwen3.5-9B/"
VISION_MODEL_VERSION="hypertok"
HYPERTOK_CKPT="/data/250010203/aisports/Codes/HyperTok/outputs/alldata_128fsq_timmvitamin_deltafilm/ckpt_epoch1.pt"
HYPERTOK_CONFIG="/data/250010203/aisports/Codes/HyperTok/configs/default.yaml"
HYPERTOK_FEATURE_SOURCE="quant"
HYPERTOK_QUANTIZER="fsq"

PROMPT_VERSION="qwen_1_5"
RUN_NAME="llavanext-hypertok-qwen35-t2i-latent"
CKPT_PATH=${LLM_VERSION}

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=$5 --nnodes=$3 --node_rank=$4 --master_addr=$1 --master_port=$2 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path /path/to/text_image_generation_data.json \
    --image_folder /path/to/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_pretrained ${HYPERTOK_CKPT} \
    --hypertok_config ${HYPERTOK_CONFIG} \
    --hypertok_feature_source ${HYPERTOK_FEATURE_SOURCE} \
    --hypertok_quantizer ${HYPERTOK_QUANTIZER} \
    --use_hypertok_tfm True \
    --enable_image_generation True \
    --image_generation_num_latents 256 \
    --image_generation_projector_type mlp2x_gelu \
    --image_generation_latent_loss_weight 1.0 \
    --image_generation_recon_loss_weight 1.0 \
    --image_generation_tune_decoder False \
    --mm_tunable_parts="image_generation_projector,mm_language_model" \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --run_name ${RUN_NAME} \
    --output_dir "/data/250010203/aisports/Codes/LLaVA-NeXT/${RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
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
    --attn_implementation sdpa \
    --qwen3_5_vl_weights True
