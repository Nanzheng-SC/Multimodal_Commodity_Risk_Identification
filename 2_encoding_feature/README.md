# 2_encoding_feature

## 目录定位

`2_encoding_feature` 负责把数据层的三类原始输入转成数值特征，并构造 `horizon30` 时间窗口。该目录对应从多模态原始数据到标准化建模输入的转换过程。

当前编码主线为：

```text
structured + text_bert + image_clip
  -> fusion (late.gru_gate)
  -> time_series_horizon30_mainline
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
│  ├─ selector.py
│  ├─ timemixer_backend.py
│  └─ run_fusion_pipeline.py
├─ time_series/
│  ├─ README.md
│  ├─ config.py
│  ├─ dataset_builder.py
│  ├─ window_builder.py
│  └─ run_time_series_pipeline.py
└─ outputs/
   └─ daily/
      ├─ structured_features/
      ├─ text_features/
      ├─ image_features/
      ├─ fusion_features/
      ├─ time_series_horizon30/
      └─ time_series_horizon30_mainline/
```

## 数据流

```text
1_data_handling
  -> structured/process_structured.py
  -> text_bert/run_text_pipeline.py
  -> image_clip/run_image_pipeline.py
  -> fusion/run_fusion_pipeline.py
  -> time_series/run_time_series_pipeline.py
  -> 3_modeling
```

## 各模块作用

### structured
- 读取 `structured_daily_merged.csv`
- 生成结构化特征、标签、索引和参考价数组
- 负责补充日频历史收益、波动率、价差和 regime 特征

### text_bert
- 使用 multilingual BERT 对文本编码
- 同一天多条文本按日分组做平均池化
- 同时保留 `text_count` 和缺失标记，便于下游识别文本覆盖强度

### image_clip
- 使用 CLIP 对图片缩略图编码
- 同一天图片按日分组做平均池化
- 输出日频图像向量、图片数和缺失标记

### fusion
- 先把结构化、文本、图像按统一索引对齐
- 再比较多种融合器并输出 canonical 融合特征
- 当前正式方法为 `late.gru_gate`

### time_series
- 基于日频特征构造滑动窗口
- 当前默认日频窗口为 `[14, 30, 90]`
- 主线窗口目录为 `time_series_horizon30_mainline`

## 关键文件说明

| 文件 | 描述性说明 |
| --- | --- |
| `config.py` | 编码层全局参数入口，统一约束文本长度、图像尺寸、批大小和共享特征维度。 |
| `run_all_real_data_pipeline.py` | 编码层总调度脚本，串联结构化、文本、图像、融合和时间窗口构建。 |
| `structured/process_structured.py` | 将结构化日表转换为建模特征和标签数组。 |
| `text_bert/run_text_pipeline.py` | 文本编码入口，读取文本主表并生成 day-level 文本向量。 |
| `image_clip/run_image_pipeline.py` | 图像编码入口，读取 manifest 和缩略图并生成日频图像向量。 |
| `fusion/modules.py` | 融合器实现文件，包括 `late.gru_gate` 等模块。 |
| `fusion/selector.py` | 融合方案选择文件，输出 canonical 融合结果。 |
| `fusion/run_fusion_pipeline.py` | 融合层入口，负责三模态索引对齐、候选融合器训练和融合向量输出。 |
| `time_series/window_builder.py` | 时间窗口构建文件，定义窗口长度、预测跨度和样本对齐。 |
| `time_series/run_time_series_pipeline.py` | 窗口构建入口，输出训练、验证和测试使用的 `X / y / index`。 |

## 关键配置

`config.py` 里当前最重要的配置有：
- `MAX_TEXT_LENGTH = 512`
- `TEXT_SUMMARY_LENGTH = 1000`
- `IMAGE_SIZE = (224, 224)`
- `LIGHT_IMAGE_SIZE = (768, 768)`
- `UNIFIED_DIM = 256`
- `BATCH_SIZE = 32`
- `DEVICE = cuda / cpu 自动判断`

频率与窗口由共享模块维护：
- 日频窗口：`[14, 30, 90]`
- 月频窗口：`[3, 6, 12]`
- 当前主线默认日频窗口：`30`

## 关键脚本与可调参数

### `run_all_real_data_pipeline.py`
统一跑完整编码流程。

常用参数：
- `--stop-after structured|text|image|fusion|time_series`
- `--frequency daily|monthly|both`

### `structured/process_structured.py`
- `--frequency daily|monthly`

### `text_bert/run_text_pipeline.py`
- `--frequency daily|monthly`

### `image_clip/run_image_pipeline.py`
- `--frequency daily|monthly`

### `fusion/run_fusion_pipeline.py`
常用参数：
- `--frequency daily|monthly`
- `--select-best`
- `--fusion-stage`
- `--fusion-method`
- `--window-length`
- `--selector-backend sequence_pool|timemixer`
- `--selection-rule val_rmse|stable_score`
- `--target-mode level|residual|relative_residual`
- `--forecast-horizon-days`
- `--gate-mode`
- `--gate-bias-init`
- `--gate-regularization-lambda`
- `--gate-max-text-share`
- `--gate-residual-scale`
- `--text-modal-dropout`
- `--image-modal-dropout`

当前融合配置：
- 默认阶段：`late`
- 默认方法：`gru_gate`
- `selection_rule = stable_score`
- `stability_lambda = 0.25`
- 文本/图像门控 dropout 分别用于抑制弱模态过拟合

### `time_series/run_time_series_pipeline.py`
常用参数：
- `--frequency daily|monthly`
- `--window-length`
- `--variants fusion,text,image,structured`

## 编码结果说明

`outputs/daily/` 下的核心产物包括：
- `structured_features/structured_features.npy`
- `text_features/text_embeddings.npy`
- `image_features/image_embeddings.npy`
- `fusion_features/fusion_features.npy`
- `time_series_horizon30*/window_xx/`

其中：
- 文本与图像都先做 item-level 编码，再按 `date` 聚合成 day-level 向量
- 融合结果是 `3_modeling` 的主输入
- 时间窗口目录把 `X / y / index` 按 train-valid-test 切分好，供模型直接读取

## 与下游的关系

- `3_modeling` 的基线模型和 TimeMixer 都从本目录的窗口结果读取输入
- `4_decision_rl` 不直接依赖编码文件，但依赖其训练后导出的正式结果
