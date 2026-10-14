from dataclasses import dataclass, field

from transformers import TrainingArguments as HfTrainingArguments


@dataclass
class TrainingArguments(HfTrainingArguments):
    negatives_cross_device: bool = field(
        default=False,
        metadata={"help": "Whether to share the negatives across all GPUs. This argument will extend the number of negatives."},
    )

    emb_loss_temperature: float = field(
        default=0.02, metadata={"help": "Temperature for softmax when computing contrastive loss."}
    )

    gen_loss_weight: float = field(
        default=0.1, metadata={"help": "Causal language modeling loss weight."}
    )

    emb_loss_weight: float = field(default=1.0, metadata={"help": "Contrasitive loss weight."})

    embed_only: bool = field(default=False, metadata={"help": "No generation; use only query and document"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Model name or local model path passed to AutoModel/AutoTokenizer.from_pretrained."}
    )

    attn_implementation: str | None = field(default=None)


@dataclass
class DataArguments:
    train_data: str = field(metadata={"help": "Train dataset path. Will be loaded with `datasets.load_from_disk` if a dir is provided, otherwise expect a jsonl file."})
    max_query_len: int = field(default=4096, metadata={"help": "Query-side truncation length."})
    max_passage_len: int = field(default=4096, metadata={"help": "Doc-side truncation length."})
    train_group_size: int = field(
        default=2, metadata={"help": "1 positive + (train_group_size - 1) negatives per query."}
    )
    preprocess_num_workers: int = field(default=4, metadata={"help": "Number of workers for preprocessing dataset."})
