PYTORCH_ALLOC_CONF=expandable_segments:True

python -m gem.run.instructir \
    --model gem \
    --checkpoint train_runs/gem_qwen3/checkpoint-500 \
    --batch_size 32 \
    --generation_batch_size 16 \
    --output_dir data/results/instructir_gem_qwen3