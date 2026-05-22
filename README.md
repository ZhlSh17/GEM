# GEM: A Generative Embedding Model Bridging Reasoning and Retrieval

## 💾 Installation
```bash
conda create -n gem python=3.10
conda activate gem

# use --recurse-submodules for running evaluation
git clone --recurse-submodules <url_of_this_anonymous_repo>
cd GEM

pip install -r requirements.txt

# [Optional] install flash attention; can be slow to build
pip install flash-attn --no-build-isolation
```

## Data Generation
### Response Generation
```bash
MODEL="Qwen/Qwen3-4B-Instruct-2507"

for DATASET in msmarco msmarco_instruct reasonir_hq;
do
    python -m gem.run.gen_resp \
        --model $MODEL \
        --dataset $DATASET \
        --output_dir data/syn
done
```

### Document Generation
```bash
for DATASET in msmarco reasonir_hq;
do
    python -m gem.run.gen_doc \
        --dataset $DATASET \
        --input_dir data/syn \
        --output_dir data/train_data
done

python -m gem.run.merge
```
The final training set will be saved at `data/train_data/all`.


## Training
Our defualt configuration runs on two GPUs, each having 48 GB of memory or more.
`WANDB_API_KEY` is required for logging.

```bash
export WANDB_API_KEY=<your_key>
bash scripts/train.sh
```

## Evaluation
```bash
# BRIGHT
bash scripts/run_bright.sh

# FollowIR
bash scripts/run_followir.sh

# InstructIR
bash scripts/run_instructir.sh
```
