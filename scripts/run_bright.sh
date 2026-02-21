PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODEL=reasonir
# REASONING=gpt4
# BS=-1

# for TASK in biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems theoremqa_questions; do
#     python -m gem.run.bright --task $TASK --model $MODEL --encode_batch_size $BS --reasoning $REASONING --concat_reasoning
# done


# biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems theoremqa_questions
# biology earth_science

MODEL=gem

for TASK in biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems theoremqa_questions ; do
    python -m gem.run.bright \
        --task $TASK \
        --model $MODEL \
        --encode_batch_size 32 \
        --generation_batch_size 16 \
        --checkpoint "train_runs/qwen3_all_v2_stage2/checkpoint-500" \
        --query_max_length 1024 \
        --doc_max_length 1024 \
        --output_dir "data/results/bright_red_all_v2_stage2" \
        --cache_dir "data/index/bright_red_all_v2_stage2" \
        # --overwrite \
        # --reasoning $REASONING \
done
