from __future__ import annotations

import numpy as np


class ImagePooling:
    def pool_group(self, group, encoder):
        image_paths = group["image_path"].tolist()
        embeddings = encoder.encode_batch(image_paths)
        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
        else:
            mean_embedding = None
        return {"embedding": mean_embedding, "count": len(embeddings)}

    def process_all_periods(self, grouped_records, encoder, index_column: str):
        results = []
        for index_value, group in grouped_records:
            result = self.pool_group(group, encoder)
            if result["embedding"] is None:
                continue
            results.append(
                {
                    index_column: index_value,
                    "embedding": result["embedding"],
                    "image_count": result["count"],
                    "image_missing_flag": 0,
                }
            )
        return results
