<h1><img src="assets/gem_logo.png" alt="logo" width="30"/> GEM: A Generative Embedding Model Bridging Reasoning and Retrieval</h1>

<p align="left">
  <a href="https://huggingface.co/gem-search/GEM-4B">🤗 Model</a> |
  <a href="https://arxiv.org/abs/2608.13200">📖 Paper</a>
</p>

GEM is a Generative Embedding Model that unifies generation and embedding within a single model. GEM augments retrieval by explicitly reasoning about user intent and relevance criteria, and it is trained to produce reasoning-aligned embeddings.


## Table of Contents
- [1. Usage](#1-usage)
    - [1.1 Load the model](#11-load-the-model)
    - [1.2 Generation](#12-generation)
    - [1.3 Embedding](#13-embedding)
- [2. Train your own GEM](#2-train-your-own-gem)
    - [2.1 Custom model](#21-custom-model)
    - [2.2 Custom data](#22-custom-data)
- [3. Reproduce](#3-reproduce)
    - [3.1 Data Generation](#31-data-generation)
    - [3.2 Training](#32-training)
    - [3.3 Evaluation](#33-evaluation)
- [Citation](#citation)


## 1. Usage
### 1.1 Load the model
```python
from transformers import AutoModel

gem = AutoModel.from_pretrained(
    "gem-search/GEM-4B",
    device_map="auto",
    trust_remote_code=True,  # required
    dtype="bfloat16",
    # attn_implementation="flash_attention_2",  # recommended
).eval()
```
Make sure to set `trust_remote_code=True` to use our custom modeling code.

### 1.2 Generation
While GEM is an embedding model, it remains generative and can be prompted just like its backbone LLM. For example:

```python
prompt = "Hi, how are you?"
tokenizer = gem.tokenizer
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer([text], return_tensors="pt").to(gem.device)
outputs = gem.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(outputs[0]))
```

### 1.3 Embedding

#### 1.3.1 Generate-then-encode
Use `generate_then_encode` (or `encode_queries`) for reasoning-augmented embedding. The idea is simple: incrementally encoding the context by appending an embedding token and reusing the KV cache from generation.

```python
query = '''"There is always a day when the flower blooms again."
Find me some similar poems.'''

# if you only need the embedding
q_embs = gem.generate_then_encode([query], max_new_tokens=512, do_sample=False)

# optionally, set `return_generation=True` to return the response
q_embs, resps = gem.generate_then_encode([query],  max_new_tokens=512, do_sample=False, return_generation=True)
print(resps[0])
```

Note that `generate_then_encode` accepts additional kwargs for generation, e.g. `do_sample`, `max_new_tokens`, and `temperature`, just like the standard `generate`.


__Alternatively, we recommend using the higher-level API, `encode_queries`, which calls `generate_then_encode` internally and supports all of its arguments, plus additional options.__
```python
query = '''"There is always a day when the flower blooms again."
Find me some similar poems.'''

q_embs = gem.encode_queries([query], max_new_tokens=512, do_sample=False)
```

#### 1.3.2 Embed-only
Use `encode_corpus` to directly encode texts without generation.
```python
doc = '''No winter stays forever.
Somewhere beneath the frost,
a quiet seed is already
dreaming of spring.'''

doc_embs = gem.encode_corpus([doc])

# all output embeddings are normalized; ready for inner product.
print(q_embs @ doc_embs.T)
```

#### 1.3.3 Use custom prompt
In GEM, embedding can be viewed as an additional skill learned by the backbone LLM; the model remains _promptable_.

By default, GEM uses a reasoning prompt. You may explicitly set your own prompt based on your tasks. Here is an example to use GEM for conventional query expansion:

```python
prompt = """# Task
Write a poem in response to the query below, using no more than 50 words.

# Query
{text}
"""

q_embs, resps = gem.encode_queries([query], prompt=prompt, max_new_tokens=128, do_sample=False, return_generation=True)

print(resps[0])
print(q_embs @ doc_embs.T)
```
PS: make sure your prompt has a `{text}` placeholder.


---
For further details, please refer to:
- `gem.model.GEM.encode_queries`
- `gem.model.GEM.encode_corpus`
- `gem.model.GEM.generate_then_encode`
- `gem.model.GEM.encode`






## 2. Train your own GEM
We provide a training script, [`scripts/train.sh`](https://github.com/ZhlSh17/GEM/blob/main/scripts/train.sh), in which you can specify your custom backbone model and training data.

### 2.1 Custom model
```bash
MODEL_NAME=<model_name_or_path>
CONFIG_FILE=<config_file_path>
```
Note: `CONFIG_FILE` is for HuggingFace's [Accelerate](https://huggingface.co/docs/transformers/en/accelerate) to configure training backends like FSDP and DeepSpeed. You can use your own config file other than those we provided in `scripts/configs`. See our [example](https://github.com/ZhlSh17/GEM/blob/main/scripts/configs/train_config_qwen3.yaml).


### 2.2 Custom data

```bash
TRAIN_DATA=<train_data_path>
```
Our training code uses HuggingFace `datasets` for data loading and we support three sources of training data:
| Source | Example |
| --- | --- |
| Local directory | `TRAIN_DATA=data/train_data/all/` |
| Local JSONL file | `TRAIN_DATA=data/train_data/all.jsonl` |
| Remote HF dataset | `TRAIN_DATA=gem-search/gem-data` |


#### Schema
The following fields are required for each data instance:
- `query` (str): Query
- `prompt` (str): A prompt template containing a `{text}` placeholder, such that `prompt.format(text=query)` constructs the input prompt
- `reason` (str, optional): An LLM-generated response, conditioned on the constructed input prompt. `reason` can be `None`, indicating an embed-only instance for which only standard contrastive loss is applied and `prompt` is ignored.
- `pos` (list[str]): A non-empty list of positive documents
- `neg` (list[str]): A non-empty list of (hard) negative documents


#### Example
```json
{
    "query": "who performs geico camel voice? ...",
    "prompt": "# Task\nIdentify the search intent. Then provide a concise analysis of the key criteria that make a document relevant.\n\n# Query\n{text}\n",
    "reason": "**Search Intent:** ... **Concise Analysis of Key Criteria for Document Relevance:**  ... those with a clear, named attribution directly tied to the Geico camel voice—should be accepted as relevant.",
    "pos": ["Who does the voice of the camel in the Geico ..."],
    "neg": ["Geico is known for its creative and memorable advertising campaigns ..."]
}
```
<!-- See our dataset [gem-search/gem-data](https://huggingface.co/datasets/gem-search/gem-data) for more examples. -->


### 2.3 Notes
Unless your goal is to reproduce our model, we recommend constructing your own dataset using stronger LLMs, more diverse queries & prompts, etc. We also recommend optimizing hyperparameters based on your setup. 

Our default configuration in [`scripts/train.sh`](https://github.com/ZhlSh17/GEM/blob/main/scripts/train.sh) works on a single node of 2 GPUs, each having >= 48 GB memory. For reference, training our [GEM-4B](https://huggingface.co/gem-search/GEM-4B) takes ~14 hours on 2 NVIDIA H100 GPUs.



## 3. Reproduce
To reproduce our results, we recommend installing dependencies as follows:
```bash
conda create -n gem python=3.10
conda activate gem

# use --recurse-submodules to clone third-party code (required only for evaluation)
git clone --recurse-submodules https://github.com/ZhlSh17/GEM.git
cd GEM

pip install -r requirements.txt

# [Optional] install flash attention
pip install flash-attn --no-build-isolation
```

### 3.1 Data Generation
Our training data was constructed using data from [ReasonIR](https://huggingface.co/datasets/reasonir/reasonir-data) and [Promptriever](https://huggingface.co/datasets/samaya-ai/msmarco-w-instructions) (msmarco).
Due to licensing and redistribution concerns, we are unable to directly release our training data. Instead, we provide scripts to reproduce it. All required source datasets will be downloaded automatically. Users are responsible for ensuring that the use of these datasets complies with their respective licenses and terms.


```bash
bash scripts/gen_data.sh
```
This script uses [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) for response generation and [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for document generation. The default vLLM configuration works on a single NVIDIA H100 GPU. The final training set will be saved to `data/train_data/all`.


### 3.2 Training
Our defualt configuration fine-tunes [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) on two GPUs, each having 48 GB of memory or more.
By default, we use WANDB for logging and `WANDB_API_KEY` is required. Alternatively, you can replace `REPORT_TO=wandb` with `REPORT_TO=tensorboard` to log locally.

```bash
export WANDB_API_KEY=<your_key>
bash scripts/train.sh
```

### 3.3 Evaluation
Run the following scripts to evaluate your trained model on BRIGHT, FollowIR, and InstructIR. The required datasets will be automatically downloaded to `data/datasets`.

```bash
# BRIGHT
bash scripts/run_bright.sh

# FollowIR
bash scripts/run_followir.sh

# InstructIR
bash scripts/run_instructir.sh
```

To directly evaluate the model we released instead of your reproduced model, specify `--checkpoint "gem-search/GEM-4B"` in the evaluation scripts above.


## Citation
```
@misc{shen2026gemgenerativeembeddingmodel,
      title={GEM: A Generative Embedding Model Bridging Reasoning and Retrieval}, 
      author={Zhili Shen and Craig Macdonald},
      year={2026},
      eprint={2608.13200},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.13200}, 
}
```