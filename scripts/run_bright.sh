PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODEL=reasonir
# BS=-1

# for TASK in biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems theoremqa_questions; do
#     python -m red.datasets.bright --task $TASK --model $MODEL --encode_batch_size $BS
# done

MODEL=reasonir
REASONING=gpt4
BS=-1

for TASK in biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems theoremqa_questions; do
    python -m red.run.bright --task $TASK --model $MODEL --encode_batch_size $BS --reasoning $REASONING --concat_reasoning
done
