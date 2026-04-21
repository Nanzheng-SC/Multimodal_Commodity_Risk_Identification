from __future__ import annotations

import numpy as np


class TextPooling:
    def pool_group(self, group, encoder):
        texts = group["text_input"].tolist()
        embeddings = encoder.encode_batch(texts)
        mean_embedding = np.mean(embeddings, axis=0)
        return {"embedding": mean_embedding, "count": len(texts)}

    def process_all_groups(self, grouped_records, index_column: str):
        monthly_results = []
        for index_value, group in grouped_records:
            result = self.pool_group(group, encoder=self.encoder)  # pragma: no cover - set by runner
            monthly_results.append(
                {
                    index_column: index_value,
                    "embedding": result["embedding"],
                    "text_count": result["count"],
                    "text_missing_flag": 0,
                }
            )
        return monthly_results

    def process_all_periods(self, grouped_records, encoder, index_column: str):
        self.encoder = encoder
        return self.process_all_groups(grouped_records, index_column)
