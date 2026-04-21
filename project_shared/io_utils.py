from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def open_text_auto(path: str | Path, mode: str = "rt", encoding: str = "utf-8"):
    target = Path(path)
    if "b" in mode:
        raise ValueError("open_text_auto only supports text modes.")
    if target.suffix == ".gz":
        return gzip.open(target, mode=mode, encoding=encoding)
    return open(target, mode=mode, encoding=encoding)


def read_jsonl_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open_text_auto(path, mode="rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl_records(path: str | Path, records: Iterable[dict]) -> Path:
    target = ensure_parent(path)
    with open_text_auto(target, mode="wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def read_env_file(path: str | Path) -> dict[str, str]:
    env: dict[str, str] = {}
    target = Path(path)
    if not target.exists():
        return env
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def dump_json(path: str | Path, payload: dict | list) -> Path:
    target = ensure_parent(path)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target

