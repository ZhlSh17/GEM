# Response sampling
MODEL="Qwen/Qwen3-4B-Instruct-2507"

for DATASET in msmarco msmarco_instruct reasonir_hq; do
    python -m gem.run.gen_resp \
        --model $MODEL \
        --dataset $DATASET \
        --output_dir data/syn
done


# Document synthesis
MODEL="meta-llama/Llama-3.1-8B-Instruct"

for DATASET in msmarco reasonir_hq; do
    python -m gem.run.gen_doc \
        --model $MODEL \
        --dataset $DATASET \
        --input_dir data/syn \
        --output_dir data/train_data
done

# Merge
python -m gem.run.merge
