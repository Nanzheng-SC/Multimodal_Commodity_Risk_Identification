import os

import numpy as np
from torch.utils.data import Dataset


class Dataset_FusionTimeSeries(Dataset):
    def __init__(self, root_path, flag="train", seq_len=12, pred_len=1):
        self.root_path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len

        if self.pred_len != 1:
            raise ValueError("FusionTimeSeries dataset currently only supports pred_len=1.")

        self._read_data()

    def _resolve_split_prefix(self) -> str:
        split_map = {
            "train": "train",
            "val": "valid",
            "test": "test",
        }
        if self.flag not in split_map:
            raise ValueError(f"Unsupported split flag: {self.flag}")
        return split_map[self.flag]

    def _read_data(self):
        split_prefix = self._resolve_split_prefix()
        window_dir = os.path.join(self.root_path, f"window_{self.seq_len}")
        feature_path = os.path.join(window_dir, f"{split_prefix}.npy")
        label_path = os.path.join(window_dir, f"{split_prefix}_labels.npy")
        month_path = os.path.join(window_dir, f"{split_prefix}_index.npy")
        if not os.path.exists(month_path):
            month_path = os.path.join(window_dir, f"{split_prefix}_months.npy")

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Cannot find fusion time-series features: {feature_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Cannot find fusion time-series labels: {label_path}")
        if not os.path.exists(month_path):
            raise FileNotFoundError(f"Cannot find fusion time-series months: {month_path}")

        self.features = np.load(feature_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.float32)
        self.months = np.load(month_path, allow_pickle=True)

        if self.features.ndim != 3:
            raise ValueError(f"Expected features shape [N, T, C], got {self.features.shape}")
        if self.features.shape[1] != self.seq_len:
            raise ValueError(
                f"Window length mismatch: dataset has {self.features.shape[1]}, expected {self.seq_len}"
            )
        if len(self.features) != len(self.labels) or len(self.features) != len(self.months):
            raise ValueError("Features, labels, and months must have the same number of samples.")

        self.seq_x_mark = np.zeros((self.seq_len, 3), dtype=np.float32)
        self.seq_y_mark = np.zeros((self.pred_len, 3), dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        seq_x = self.features[idx]
        seq_y = np.asarray([[self.labels[idx]]], dtype=np.float32)
        return seq_x, seq_y, self.seq_x_mark.copy(), self.seq_y_mark.copy()

    def get_dates(self):
        return self.months
