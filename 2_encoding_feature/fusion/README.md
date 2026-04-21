# fusion

本目录负责对齐结构化、文本和图片三模态特征，并生成建模使用的多模态融合特征。当前最终主线融合器为 `late.gru_gate`，结果进入日频 horizon30 TimeMixer。

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `align_and_merge.py` | 读取 `outputs/daily/{structured,text,image}_features/`，按 `date` 对齐三模态特征。 |
| `config.py` | 融合方法、输出维度、选择器训练参数和路径配置。 |
| `modules.py` | early、intermediate、late 三类融合模块实现，包含 `late.gru_gate`。 |
| `selector.py` | 融合器选择、稳定性评分、canonical 特征同步。 |
| `timemixer_backend.py` | 用轻量 TimeMixer 后端评估融合表示。 |
| `run_fusion_pipeline.py` | 融合入口脚本。 |

## 输入

```text
2_encoding_feature/outputs/daily/structured_features/
2_encoding_feature/outputs/daily/text_features/
2_encoding_feature/outputs/daily/image_features/
```

关键输入文件包括 `structured_features.npy`、`text_embeddings.npy`、`image_embeddings.npy`、各自的 `*_index.npy` 和缺失标记。

## 输出

```text
2_encoding_feature/outputs/daily/fusion_features/
├─ fusion_features.npy
├─ fusion_features_daily.csv
├─ fusion_labels.npy
├─ fusion_index.npy
├─ fusion_months.npy
├─ fusion_missing_flags.npy
├─ fusion_selection_results.csv
├─ best_fusion_choice.json
├─ metrics.json
└─ scaler_params.npz
```

`fusion_missing_flags.npy` 用于确认文本和图片日频覆盖是否完整；当前主线目标是文本/图片缺失为 0。

## 可调整参数

| 参数位置 | 影响 |
| --- | --- |
| `config.py` 的融合方法 | 决定参与评估和同步的 fusion 方法。 |
| `fused_dim` | 控制融合特征维度，影响表达能力和训练速度。 |
| `selector_backend` | 控制融合器选择的验证后端。 |
| `gate_bias_init` / `modal_dropout` | 控制 `late.gru_gate` 对文本和图片信号的初始偏好与训练鲁棒性。 |

## 运行

```powershell
python 2_encoding_feature/fusion/run_fusion_pipeline.py --frequency daily --select-best --selector-backend timemixer
```

如果只需要重建当前主线，可显式指定 `late.gru_gate`，再运行 time-series 窗口构建。
