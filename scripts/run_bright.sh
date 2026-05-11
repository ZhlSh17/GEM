PYTORCH_ALLOC_CONF=expandable_segments:True


for TASK in biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_questions  theoremqa_theorems ; do
    python -m gem.run.bright \
        --task $TASK \
        --model gem \
        --encode_batch_size 32 \
        --checkpoint "train_runs/gem_qwen3/checkpoint-500" \
        --query_max_length 1024 \
        --doc_max_length 1024 \
        --output_dir "data/results/bright_gem_qwen3" \
        --cache_dir "data/index/bright_gem_qwen3" \
        --generation_batch_size 16
done
