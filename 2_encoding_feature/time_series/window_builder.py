from __future__ import annotations

import numpy as np

from .config import TIME_SERIES_CONFIG


class WindowBuilder:
    def __init__(self, window_lengths=None, default_window_length=None):
        self.window_lengths = list(window_lengths or TIME_SERIES_CONFIG["window_lengths"])
        self.default_window_length = int(default_window_length or TIME_SERIES_CONFIG["default_window_length"])

    def build_windows(self, features, labels, window_length=None):
        if window_length is None:
            window_length = self.default_window_length
        if window_length not in self.window_lengths:
            raise ValueError(f"Window length must be one of {self.window_lengths}")

        X = []
        y = []
        for start in range(0, len(features) - window_length + 1):
            end = start + window_length
            window_features = features[start:end]
            window_label = labels[end - 1]
            if not np.isnan(window_label) and not np.isnan(window_features).any():
                X.append(window_features)
                y.append(window_label)
        return np.array(X), np.array(y)

    def build_windows_with_months(self, features, labels, months, window_length=None):
        if window_length is None:
            window_length = self.default_window_length

        X = []
        y = []
        window_months = []
        for start in range(0, len(features) - window_length + 1):
            end = start + window_length
            window_features = features[start:end]
            window_label = labels[end - 1]
            window_month = months[end - 1]
            if not np.isnan(window_label) and not np.isnan(window_features).any():
                X.append(window_features)
                y.append(window_label)
                window_months.append(window_month)
        return np.array(X), np.array(y), np.array(window_months)
