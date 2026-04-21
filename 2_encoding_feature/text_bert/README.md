# text_bert

本目录负责把真实文本主表编码成日频文本特征。当前默认使用本地 `bert-base-multilingual-cased`，如果本地模型不存在则回退到 HuggingFace 模型名。

## 文件结构

| 文件或目录 | 作用 |
| --- | --- |
| `config.py` | 文本模型、batch size、最大长度和输出路径配置。 |
| `dataset.py` | 读取 gzip JSONL 文本主表，拼接 `title`、`summary` 和可用 `text`。 |
| `encoder.py` | 标准 multilingual BERT 编码器。 |
| `encoder_local.py` | 本地模型优先的编码器。 |
| `pooling.py` | 同一天多条文本的池化逻辑。 |
| `run_text_pipeline.py` | 文本编码入口。 |
| `download_multilingual_model.py` | 下载或补充本地 multilingual BERT 权重。 |
| `models/` | 本地模型权重目录，权重大文件由 `.gitignore` 排除。 |

## 输入

```text
1_data_handling/raw/text/text_documents_multisource_cleaned.jsonl.gz
2_encoding_feature/outputs/daily/structured_features/structured_index.npy
```

文本主表覆盖 `1461/1461` 天，包含新闻、官方能源文本和公开日频文本。

## 输出

```text
2_encoding_feature/outputs/daily/text_features/
├─ text_embeddings.npy
├─ text_features_daily.csv
├─ text_counts.npy
├─ text_missing_flags.npy
├─ text_index.npy
└─ text_months.npy
```

同一天多条文本会先逐条编码，再按日期池化；`text_counts.npy` 记录每天文本数量，`text_missing_flags.npy` 用于确认日频覆盖。

## 可调整参数

| 参数 | 影响 |
| --- | --- |
| `BERT_MODEL_NAME` | 更换文本语义模型。 |
| `MAX_TEXT_LENGTH` | 控制单条文本截断长度，影响上下文保留和速度。 |
| `TEXT_SUMMARY_LENGTH` | 控制摘要拼接长度。 |
| `BATCH_SIZE` | 控制吞吐和显存。 |
| pooling 策略 | 影响同一天多条文本如何汇总成一个日频向量。 |

## 运行

```powershell
python 2_encoding_feature/text_bert/run_text_pipeline.py --frequency daily
```

重跑文本编码后，需要继续重跑 `fusion/run_fusion_pipeline.py` 和 `time_series/run_time_series_pipeline.py`。
