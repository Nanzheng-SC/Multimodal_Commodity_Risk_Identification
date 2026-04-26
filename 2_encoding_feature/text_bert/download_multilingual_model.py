
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


MODEL_ID = "bert-base-multilingual-cased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download multilingual BERT for offline encoding.")
    parser.add_argument("--project-root", default=".", help="Project root path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    target_dir = project_root / "2_encoding_feature" / "text_bert" / "models" / MODEL_ID
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[model-download] target_dir={target_dir}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    print(f"[model-download] completed model_id={MODEL_ID}")


if __name__ == "__main__":
    main()
