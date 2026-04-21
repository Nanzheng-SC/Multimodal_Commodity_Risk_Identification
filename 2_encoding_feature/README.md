# 2_encoding_feature

## 目录定位

`2_encoding_feature` 负责把数据层的三类原始输入转成可建模的数值特征，并进一步构造 `horizon30` 时间窗口。这里完成的是“从原始多模态数据到标准化建模输入”的转换。

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

## 数据如何流动

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
| `config.py` | 这是编码层的全局参数入口，统一约束文本长度、图像尺寸、批大小和共享特征维度。只要这里调整，文本、图像和融合编码的尺寸与资源占用都会随之变化。 |
| `run_all_real_data_pipeline.py` | 这是编码层总调度脚本，用来把结构化、文本、图像、融合和时间窗口串成一条可重复执行的流水线。它最重要的作用是保证不同模态的输出口径始终一致。 |
| `structured/process_structured.py` | 该文件负责把结构化日表转成建模用的数值特征和标签数组。收益率、波动率、价差和 regime 特征的构造都在这里完成，因此它决定了结构化模态的表达质量。 |
| `text_bert/run_text_pipeline.py` | 这是文本编码主入口，负责读取文本主表、调用 multilingual BERT、并把同日多条文本池化成 day-level 向量。它决定了文本模态是以“单篇”还是“按日聚合”的形式进入后续建模。 |
| `image_clip/run_image_pipeline.py` | 这是图像编码主入口，负责读取 manifest、加载缩略图并生成日频图像向量。它把图片文件系统组织转换成模型可消费的统一 embedding。 |
| `fusion/modules.py` | 这里实现了融合器本体，包括 `late.gru_gate` 等关键模块。它不只是定义层结构，还决定了文本、图像与结构化之间怎样竞争和协同。 |
| `fusion/selector.py` | 该文件负责在多种融合方案之间做正式选择，并输出 canonical 融合结果。也就是说，最终进入主线建模的融合特征由这里裁定。 |
| `fusion/run_fusion_pipeline.py` | 这是融合层总入口，负责对齐三模态索引、训练候选融合器并产出最终融合向量。主线 `late.gru_gate` 的正式特征就是通过它生成的。 |
| `time_series/window_builder.py` | 这个文件把连续日频特征转成监督学习窗口，是从“静态特征矩阵”走向“时间序列样本”的关键一步。窗口长度、预测跨度和对齐方式都在这一层落实。 |
| `time_series/run_time_series_pipeline.py` | 这是窗口构建入口，负责输出训练、验证、测试可直接读取的 `X / y / index`。它把编码层和建模层真正接了起来。 |

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

当前融合配置的核心点是：
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
