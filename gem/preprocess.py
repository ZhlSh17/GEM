from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, PreTrainedModel


EMBED_TOKEN = "<|embed|>"

QUERY_INSTRUCTION = (
    "Identify the search intent. Then provide a concise analysis of the key criteria that make a document relevant."
)

PASSAGE_INSTRUCTION = "Represent the text for retrieval."

_QUERY_PROMPT = """\
# Task
{instruction}

# Query
{text}
"""

_PASSAGE_PROMPT = """\
# Task
{instruction}

# Text
{text}
"""


class Preprocessor:

    def __init__(self, tokenizer: "PreTrainedTokenizerBase") -> None:
        if not tokenizer.chat_template:
            raise NotImplementedError("expect chat model")

        self.tokenizer = tokenizer

    def preprocess_passage(
        self,
        passage: str,
        *,
        instruction: str | None = PASSAGE_INSTRUCTION,
        append_embed_token: bool = True,
    ) -> str:
        if instruction is None:
            instruction = PASSAGE_INSTRUCTION

        conversation = [
            {
                "role": "user",
                "content": _PASSAGE_PROMPT.format(text=passage, instruction=instruction),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        if append_embed_token:
            prompt += EMBED_TOKEN

        return prompt

    def preprocess_query(
        self,
        query: str,
        *,
        instruction: str | None = QUERY_INSTRUCTION,
        reason: str | None = None,
        append_embed_token: bool = True,
    ) -> str:
        if instruction is None:
            instruction = QUERY_INSTRUCTION

        conversation = [
            {
                "role": "user",
                "content": _QUERY_PROMPT.format(text=query, instruction=instruction),
            },
        ]

        add_generation_prompt = False
        if reason is None:
            # for generation mode
            add_generation_prompt = True
            append_embed_token = False
        else:
            conversation.append({"role": "assistant", "content": reason})

        prompt = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

        if append_embed_token:
            prompt += EMBED_TOKEN

        return prompt


def add_emb_token(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
) -> None:
    if EMBED_TOKEN in tokenizer.vocab:
        logger.info(f"special token {EMBED_TOKEN} already added to vocab")
        vocab_size = len(tokenizer)
        embedding_size = model.get_input_embeddings().num_embeddings
        if vocab_size != embedding_size:
            raise RuntimeError(
                f"inconsistent {vocab_size=} and {embedding_size=}\n"
                f"since we don't pad embedding matrix when resize, make sure {EMBED_TOKEN} "
                "doesn't conflict with existing tokens in the vocab"
            )

        return

    num_added = tokenizer.add_tokens([EMBED_TOKEN], special_tokens=True)
    if num_added > 0:
        logger.info("resize token embeddings")
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f"special token {EMBED_TOKEN} added to vocab")


def ensure_emb_token_is_added(tokenizer: "PreTrainedTokenizerBase") -> None:
    if EMBED_TOKEN not in tokenizer.vocab:
        raise RuntimeError(f"{EMBED_TOKEN=} not found in vocab")
