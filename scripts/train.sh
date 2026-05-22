export NCCL_P2P_DISABLE=1

MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
CONFIG_FILE="scripts/configs/train_config_qwen3.yaml"

TRAIN_DATA="data/train_data/all"
OUTPUT_DIR="train_runs/gem_qwen3"
RUN_NAME=gem_qwen3

# REPORT_TO="none"
# REPORT_TO="tensorboard"
REPORT_TO="wandb"  # require WANDB_API_KEY 
export WANDB_PROJECT="GEM"

PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=32
N_GPUS=2

MAX_STEPS=500
SAVE_STEPS=250
SAVE_TOTAL_LIMIT=1
# RESUME_FROM_CHECKPOINT=$OUTPUT_DIR/checkpoint-250



accelerate launch \
    --config_file $CONFIG_FILE \
    --num_processes $N_GPUS \
    \
    -m gem.run.train \
    --train_data $TRAIN_DATA \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    \
    --max_query_len 1024 \
    --max_passage_len 1024 \
    --train_group_size 2 \
    --preprocess_num_workers 8 \
    --dataloader_drop_last \
    \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 50 \
    --bf16 \
    \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --gradient_checkpointing \
    \
    --negatives_cross_device \
    --emb_loss_temperature 0.02 \
    --gen_loss_weight 0.1 \
    --emb_loss_weight 1.0 \
    \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --save_steps $SAVE_STEPS \
    --logging_steps 1 \
    --run_name $RUN_NAME \
    --report_to $REPORT_TO \
    # --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
    # --attn_implementation flash_attention_2
