# time_series

本目录负责把日频特征矩阵转换成建模可直接读取的固定长度滑动窗口。当前主线任务是日频 horizon30，输入变体包括 `fusion`、`text`、`image`、`structured`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `config.py` | 窗口长度、切分比例和输出目录配置。 |
| `dataset_builder.py` | 组织特征、标签、reference 和索引。 |
| `window_builder.py` | 构建固定长度滑动窗口。 |
| `run_time_series_pipeline.py` | 时间窗口生成入口。 |

## 输入

```text
2_encoding_feature/outputs/daily/fusion_features/
2_encoding_feature/outputs/daily/text_features/
2_encoding_feature/outputs/daily/image_features/
2_encoding_feature/outputs/daily/structured_features/
```

## 输出

当前活动目录为：

```text
2_encoding_feature/outputs/daily/time_series_horizon30/
├─ structured/
├─ text/
└─ image/

2_encoding_feature/outputs/daily/time_series_horizon30_mainline/
└─ late_gru_gate/
   ├─ window_90/
   └─ window_120/
```

每个窗口目录通常包含：

| 文件 | 作用 |
| --- | --- |
| `train.npy` / `valid.npy` / `test.npy` | 窗口特征张量。 |
| `train_labels.npy` / `valid_labels.npy` / `test_labels.npy` | 对应监督标签。 |
| `train_index.npy` / `valid_index.npy` / `test_index.npy` | 日期索引。 |
| `train_months.npy` / `valid_months.npy` / `test_months.npy` | 月份索引。 |

## 可调整参数

| 参数 | 当前口径 | 影响 |
| --- | --- | --- |
| 日频窗口长度 | `[14, 30, 90]`，主线实际保留 `90/120` | 控制模型可见历史长度。 |
| 输入变体 | `fusion,text,image,structured` | 控制单模态与多模态比较。 |
| target horizon | `30` | 控制未来均价残差的预测跨度。 |
| 切分策略 | 固定尾部验证 / 测试 + rolling | 影响模型选择和最终评估一致性。 |

## 运行

```powershell
python 2_encoding_feature/time_series/run_time_series_pipeline.py --frequency daily --variants fusion,text,image,structured
```

如果只修改建模超参数，不需要重建 time-series；如果修改窗口长度、输入特征或任务 horizon，则需要重建。

## 当前作用

time-series 层把三模态编码结果转换成统一训练样本，是 `3_modeling` 的直接上游。当前主线已经保留了最终需要的 unimodal 窗口和 `late_gru_gate` 主线窗口，因此后续建模和前端联调都可以直接从这里接续。
