PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
TRAIN_DATA="data/train_data/all"
OUTPUT_DIR="train_runs/qwen3_new_data"
# RESUME_FROM_CHECKPOINT=$OUTPUT_DIR/checkpoint-500

# REPORT_TO="none"
# REPORT_TO="tensorboard"
REPORT_TO="wandb"  # require WANDB_API_KEY 
export WANDB_PROJECT="RED"
RUN_NAME=$MODEL_NAME

MAX_STEPS=500
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=2
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=32
N_GPUS=2


accelerate launch \
    --config_file scripts/configs/train_config_qwen3.yaml \
    --num_processes $N_GPUS \
    \
    -m red.run.train \
    --train_data $TRAIN_DATA \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --attn_implementation flash_attention_2 \
    \
    --max_query_len 1536 \
    --max_passage_len 1024 \
    --train_group_size 2 \
    --preprocess_num_workers 8 \
    --dataloader_drop_last \
    \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
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
