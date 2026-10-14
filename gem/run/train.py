import os

from datasets import load_from_disk, load_dataset
from transformers import set_seed, HfArgumentParser, AutoTokenizer, AutoConfig
from loguru import logger

from gem.train.data import GEMDataCollator, GEMDataset
from gem.model import GEM
from gem.preprocess import add_emb_token
from gem.train.trainer import GEMTrainer
from gem.train.args import ModelArguments, DataArguments, TrainingArguments


def _parse_args() -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, train_args


def main() -> None:
    # Args
    model_args, data_args, train_args = _parse_args()

    if (
        os.path.exists(train_args.output_dir)
        and os.listdir(train_args.output_dir)
        and train_args.resume_from_checkpoint is None
        and not train_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory {train_args.output_dir} already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    if train_args.gradient_checkpointing:
        train_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    set_seed(train_args.seed)

    # Model
    model = GEM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=model_args.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    add_emb_token(model, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"set pad_token to eos_token: {tokenizer.pad_token}")

    # Data
    if os.path.isdir(data_args.train_data):
        _dataset = load_from_disk(data_args.train_data)
    elif os.path.isfile(data_args.train_data) and data_args.train_data.endswith(".jsonl"):
        _dataset = load_dataset("json", data_files=data_args.train_data)["train"]
    else:
        _dataset = load_dataset(data_args.train_data)["train"]

    train_dataset = GEMDataset(
        dataset=_dataset,
        tokenizer=tokenizer,
        max_query_len=data_args.max_query_len,
        max_passage_len=data_args.max_passage_len,
        train_group_size=data_args.train_group_size,
        num_proc=data_args.preprocess_num_workers,
        embed_only=train_args.embed_only,
    )
    data_collator = GEMDataCollator(tokenizer=tokenizer)

    trainer = GEMTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        # args for loss
        negatives_cross_device=train_args.negatives_cross_device,
        emb_loss_temperature=train_args.emb_loss_temperature,
        gen_loss_weight=train_args.gen_loss_weight,
        emb_loss_weight=train_args.emb_loss_weight,
    )

    # assume sharing cache dir in multi-node setting
    with trainer.accelerator.main_process_first():
        # the other processes will load from the cache
        train_dataset.preprocess()

    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
