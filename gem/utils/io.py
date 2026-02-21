import os
import json
from pathlib import Path
from collections.abc import Iterator


ROOT_DIR = Path(__file__).absolute().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
HF_DATASETS_DIR = DATA_DIR / "datasets"


def read_last_line(file_path: str | bytes | os.PathLike, encoding: str = "utf-8") -> str:
    """Return the last line (including trailing newline if exists) of a file."""
    with open(file_path, "rb") as f:
        file_size = f.seek(0, os.SEEK_END)
        if file_size == 0:
            return ""

        pos = f.seek(file_size - 1)
        if f.read(1) == b"\n":
            pos -= 1

        while pos >= 0:
            f.seek(pos)
            if f.read(1) == b"\n":
                pos += 1
                break
            pos -= 1

        f.seek(pos if pos > 0 else 0)
        return f.readline().decode(encoding)


def load_jsonl(file_path: str | bytes | os.PathLike, encoding: str = "utf-8") -> list[dict]:
    data = []
    with open(file_path, "r", encoding=encoding) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_jsonl_iter(file_path: str | bytes | os.PathLike, encoding: str = "utf-8") -> Iterator[dict]:
    with open(file_path, "r", encoding=encoding) as f:
        for line in f:
            yield json.loads(line)


def save_jsonl(data: list[dict], file_path: str | bytes | os.PathLike, encoding: str = "utf-8") -> None:
    with open(file_path, "wt", encoding=encoding) as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
