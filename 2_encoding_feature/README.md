# 2_encoding_feature：三模态编码与时间窗口层

本目录把 `1_data_handling` 中的结构化、文本和图片数据转成模型可直接读取的数值特征，并进一步构造日频 horizon30 时间窗口。当前主线输出是：

```text
fusion + late.gru_gate + horizon30 daily windows
```

## 目录结构

```text
2_encoding_feature/
├─ README.md
├─ config.py
├─ run_all_real_data_pipeline.py
├─ structured/
│  ├─ README.md
│  ├─ __init__.py
│  └─ process_structured.py
├─ text_bert/
│  ├─ README.md
│  ├─ config.py
│  ├─ dataset.py
│  ├─ download_multilingual_model.py
│  ├─ encoder.py
│  ├─ encoder_local.py
│  ├─ pooling.py
│  ├─ run_text_pipeline.py
│  └─ models/
├─ image_clip/
│  ├─ README.md
│  ├─ config.py
│  ├─ dataset.py
│  ├─ encoder.py
│  ├─ encoder_local.py
│  ├─ pooling.py
│  └─ run_image_pipeline.py
├─ fusion/
│  ├─ README.md
│  ├─ align_and_merge.py
│  ├─ config.py
│  ├─ modules.py
│  ├─ run_fusion_pipeline.py
│  ├─ selector.py
│  └─ timemixer_backend.py
├─ time_series/
│  ├─ README.md
│  ├─ config.py
│  ├─ dataset_builder.py
│  ├─ run_time_series_pipeline.py
│  └─ window_builder.py
└─ outputs/
   └─ daily/
      ├─ structured_features/
      ├─ text_features/
      ├─ image_features/
      ├─ fusion_features/
      ├─ time_series_horizon30/
      └─ time_series_horizon30_mainline/
```

## 文件作用

| 文件或目录 | 作用 |
| --- | --- |
| `config.py` | 编码层全局配置，统一输入路径、模型名称、设备、窗口长度和批大小。 |
| `run_all_real_data_pipeline.py` | 编码总入口，可按 structured、text、image、fusion 阶段顺序执行。 |
| `structured/process_structured.py` | 读取结构化主表，生成标准化特征、索引、reference 和结构化标签。 |
| `text_bert/dataset.py` | 文本 JSONL 读取、字段拼接和批处理。 |
| `text_bert/encoder.py` / `encoder_local.py` | multilingual BERT 文本编码实现。 |
| `text_bert/pooling.py` | 同一天多条文本的池化逻辑。 |
| `text_bert/run_text_pipeline.py` | 文本编码入口。 |
| `text_bert/download_multilingual_model.py` | 补充本地 multilingual BERT 权重。 |
| `image_clip/dataset.py` | 图片 manifest 与缩略图读取。 |
| `image_clip/encoder.py` / `encoder_local.py` | CLIP 图像编码实现。 |
| `image_clip/pooling.py` | 同一天多张图片的池化逻辑。 |
| `image_clip/run_image_pipeline.py` | 图像编码入口。 |
| `fusion/align_and_merge.py` | 按 `date` 对齐 structured、text、image 三模态。 |
| `fusion/modules.py` | 多类融合器实现，当前主线为 `late.gru_gate`。 |
| `fusion/selector.py` | 融合方法评估、选择和主线同步。 |
| `fusion/timemixer_backend.py` | 用轻量 TimeMixer 后端评估融合表示。 |
| `fusion/run_fusion_pipeline.py` | 融合特征生成入口。 |
| `time_series/window_builder.py` | 固定长度滑动窗口构建。 |
| `time_series/dataset_builder.py` | 组织 train/valid/test 样本、标签和索引。 |
| `time_series/run_time_series_pipeline.py` | horizon30 窗口生成入口。 |
| `outputs/daily/` | 当前主线编码结果目录。 |

## 数据如何流动

1. `structured/process_structured.py` 读取 `1_data_handling/raw/structured/structured_daily_merged.csv`，生成结构化特征矩阵。
2. `text_bert/run_text_pipeline.py` 读取 `text_documents_multisource_cleaned.jsonl.gz`，编码后按天池化成日频文本向量。
3. `image_clip/run_image_pipeline.py` 读取 `image_manifest_commons_cleaned.jsonl.gz` 和 `thumbnail_path`，编码后按天池化成日频图像向量。
4. `fusion/run_fusion_pipeline.py` 对齐三模态日期，生成最终 `fusion_features.npy`。
5. `time_series/run_time_series_pipeline.py` 基于 fusion 或单模态特征构造 horizon30 窗口，供 `3_modeling` 直接读取。

## 仓库同步说明

`2_encoding_feature/outputs/` 中的 `.npy`、`.csv`、`.npz` 和窗口样本文件均属于编码派生产物，体积较大，统一由本目录脚本在本地生成，不作为 GitHub 仓库内容上传。

对应生成入口为：

```powershell
python 2_encoding_feature/run_all_real_data_pipeline.py
python 2_encoding_feature/time_series/run_time_series_pipeline.py --frequency daily --variants fusion,text,image,structured
```

因此，仓库保留的是编码逻辑、配置和说明文档；实际编码结果由运行代码后自动落盘。

## 当前主线输出

### 结构化特征

```text
2_encoding_feature/outputs/daily/structured_features/
```

关键产物：

| 文件 | 作用 |
| --- | --- |
| `structured_daily_processed.csv` | 结构化处理后的可读表。 |
| `structured_features.npy` | 结构化特征矩阵，当前形状为 `(1461, 22)`。 |
| `structured_labels.npy` | 与结构化特征对齐的标签数组。 |
| `structured_reference.npy` | Brent reference 序列。 |
| `structured_index.npy` | 日期索引。 |
| `structured_feature_manifest.json` | 结构化特征字段说明。 |

### 文本特征

```text
2_encoding_feature/outputs/daily/text_features/
```

关键产物：

| 文件 | 作用 |
| --- | --- |
| `text_embeddings.npy` | 日频文本向量，当前形状为 `(1461, 768)`。 |
| `text_features_daily.csv` | 可读文本特征表。 |
| `text_counts.npy` | 每日文本条数。 |
| `text_missing_flags.npy` | 文本缺失标记。 |
| `text_index.npy` | 日期索引。 |

### 图像特征

```text
2_encoding_feature/outputs/daily/image_features/
```

关键产物：

| 文件 | 作用 |
| --- | --- |
| `image_embeddings.npy` | 日频图像向量，当前形状为 `(1461, 768)`。 |
| `image_features_daily.csv` | 可读图像特征表。 |
| `image_counts.npy` | 每日图片条数。 |
| `image_missing_flags.npy` | 图片缺失标记。 |
| `image_index.npy` | 日期索引。 |

### 融合特征

```text
2_encoding_feature/outputs/daily/fusion_features/
```

关键产物：

| 文件 | 作用 |
| --- | --- |
| `fusion_features.npy` | 最终融合特征矩阵，当前形状为 `(1461, 512)`。 |
| `fusion_features_daily.csv` | 可读融合特征表。 |
| `fusion_labels.npy` | 与融合特征对齐的标签。 |
| `fusion_missing_flags.npy` | 模态缺失标记，当前主线用于正式训练时为全覆盖。 |
| `fusion_index.npy` | 日期索引。 |
| `best_fusion_choice.json` | 融合方法选择结果。 |
| `fusion_selection_results.csv` | 融合方法评估明细。 |

当前 `best_fusion_choice.json` 显示：

- `selected_method = late.gru_gate`
- `selector_backend = timemixer`
- `target_mode = residual`
- `forecast_horizon_days = 30`
- `output_feature_dim = 512`

### 时间窗口

```text
2_encoding_feature/outputs/daily/time_series_horizon30_mainline/late_gru_gate/
```

当前主线保留 `window_90` 和 `window_120` 两组窗口。以 `window_90` 为例：

| 文件 | 形状 | 作用 |
| --- | --- | --- |
| `train.npy` | `(1222, 90, 512)` | 训练窗口特征。 |
| `valid.npy` | `(60, 90, 512)` | 验证窗口特征。 |
| `test.npy` | `(60, 90, 512)` | 测试窗口特征。 |
| `train_labels.npy` | `(1222,)` | 训练标签。 |
| `valid_labels.npy` | `(60,)` | 验证标签。 |
| `test_labels.npy` | `(60,)` | 测试标签。 |

## 可调整脚本与参数

### 顶层入口

| 脚本 | 常用参数 | 影响 |
| --- | --- | --- |
| `run_all_real_data_pipeline.py` | `--stages structured,text,image,fusion`、`--frequency daily` | 控制编码执行阶段。 |
| `config.py` | 输入路径、设备、batch size、窗口集合、模型名称 | 影响整个编码层的基础口径。 |

### 文本编码

| 脚本或参数 | 作用 | 影响 |
| --- | --- | --- |
| `text_bert/run_text_pipeline.py` | 执行文本编码 | 控制文本编码是否刷新。 |
| `BERT_MODEL_NAME` | 指定 multilingual BERT 模型 | 影响文本语义表达能力。 |
| `MAX_TEXT_LENGTH` | 文本截断长度 | 影响速度和上下文保留。 |
| `BATCH_SIZE` | 编码批大小 | 影响吞吐和显存。 |

### 图像编码

| 脚本或参数 | 作用 | 影响 |
| --- | --- | --- |
| `image_clip/run_image_pipeline.py` | 执行图像编码 | 刷新图片特征。 |
| `CLIP_MODEL_NAME` | 指定 CLIP 模型 | 影响图像表达质量和速度。 |
| `IMAGE_SIZE` | 输入尺寸 | 影响速度和细节保留。 |

### 融合与窗口

| 脚本或参数 | 作用 | 影响 |
| --- | --- | --- |
| `fusion/run_fusion_pipeline.py` | 选择并导出融合特征 | 决定主线融合表示。 |
| `fusion/config.py` | 融合维度、选择后端、融合方法列表 | 影响融合输出和比较口径。 |
| `time_series/run_time_series_pipeline.py` | 生成建模窗口 | 决定下游训练样本形状。 |
| `WINDOW_LENGTHS_BY_FREQUENCY` | 日频窗口集合 | 影响模型可见历史长度。 |

## 当前编码层结果解读

当前编码层已经完成三件关键工作：

- 三模态在 `1461` 个自然日上完成对齐
- 融合层已经稳定选出 `late.gru_gate`
- 主线建模窗口已经生成，`3_modeling` 可直接读取，无需再回到数据层二次清洗

这意味着当前项目已经形成完整的多模态日频特征链路，编码层本身不会成为后续建模和前端联调的阻塞点。
