import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from PIL import Image
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from .config import IMAGE_ENCODER_CONFIG


class ImageEncoder:
    def __init__(self):
        self.model_name = IMAGE_ENCODER_CONFIG["model_name"]
        self.fallback_model_names = IMAGE_ENCODER_CONFIG.get("fallback_model_names", [])
        self.image_size = IMAGE_ENCODER_CONFIG["image_size"]
        self.batch_size = IMAGE_ENCODER_CONFIG["batch_size"]
        self.device = IMAGE_ENCODER_CONFIG["device"]

        self.loaded_model_name = None
        self.processor = None
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
                self.processor = CLIPProcessor.from_pretrained(candidate, local_files_only=True)
                self.model = CLIPModel.from_pretrained(candidate, local_files_only=True).to(self.device)
                self.loaded_model_name = candidate
                if candidate == self.model_name:
                    print(f"[ImageEncoder] Loaded local model cache: {candidate}")
                else:
                    print(
                        f"[ImageEncoder] Local cache for {self.model_name} not found. "
                        f"Fallback to {candidate}."
                    )
                return
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

        raise RuntimeError(
            "No usable local image encoder model was found. "
            "Please cache the configured model or add a local fallback.\n"
            + "\n".join(errors)
        )

    def encode_batch(self, image_paths):
        embeddings = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images = []
            for path in batch_paths:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception as exc:
                    print(f"Error loading image {path}: {exc}")

            if not images:
                continue

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    batch_embeddings = outputs.last_hidden_state.cpu().numpy()
                else:
                    batch_embeddings = outputs.cpu().numpy()
                if batch_embeddings.ndim == 3:
                    batch_embeddings = np.mean(batch_embeddings, axis=1)
                embeddings.extend(batch_embeddings)

        return embeddings

    def encode_single(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    embedding = outputs.last_hidden_state.cpu().numpy()[0]
                else:
                    embedding = outputs.cpu().numpy()[0]
                if embedding.ndim == 2:
                    embedding = np.mean(embedding, axis=0)
                return embedding
        except Exception as exc:
            print(f"Error encoding image {image_path}: {exc}")
            return None
