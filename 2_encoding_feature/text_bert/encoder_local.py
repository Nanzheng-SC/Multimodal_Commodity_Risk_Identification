
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModel, AutoTokenizer

from .config import TEXT_ENCODER_CONFIG


class TextEncoder:
    def __init__(self):
        self.model_name = TEXT_ENCODER_CONFIG["model_name"]
        self.fallback_model_names = TEXT_ENCODER_CONFIG.get("fallback_model_names", [])
        self.max_length = TEXT_ENCODER_CONFIG["max_length"]
        self.batch_size = TEXT_ENCODER_CONFIG["batch_size"]
        self.device = TEXT_ENCODER_CONFIG["device"]

        self.loaded_model_name = None
        self.tokenizer = None
        self.model = None
        self._load_model()
        self.model.eval()

    def _candidate_model_names(self):
        candidates = [self.model_name]
        for name in self.fallback_model_names:
            if name and name not in candidates:
                candidates.append(name)
        return candidates

    def _load_model(self):
        errors = []
        for candidate in self._candidate_model_names():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
                self.model = AutoModel.from_pretrained(candidate, local_files_only=True).to(self.device)
                self.loaded_model_name = candidate
                if candidate == self.model_name:
                    print(f"[TextEncoder] Loaded local model cache: {candidate}")
                else:
                    print(
                        f"[TextEncoder] Local cache for {self.model_name} not found. "
                        f"Fallback to {candidate}."
                    )
                return
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

        raise RuntimeError(
            "No usable local text encoder model was found. "
            "Please cache the configured model or add a local fallback.\n"
            + "\n".join(errors)
        )

    def encode_batch(self, texts):
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                embeddings.extend(batch_embeddings)

        return embeddings

    def encode_single(self, text):
        return self.encode_batch([text])[0]


if __name__ == "__main__":
    encoder = TextEncoder()
    test_texts = ["This is a test sentence.", "这是一个中文测试句子。"]
    embeddings = encoder.encode_batch(test_texts)
    print(f"Encoded {len(embeddings)} texts")
    print(f"Embedding shape: {embeddings[0].shape}")
