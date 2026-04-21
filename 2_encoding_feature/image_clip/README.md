# image_clip

本目录负责把真实图片 manifest 中的 WebP 缩略图编码成日频图片特征。当前主线只读取 `thumbnail_path`，不依赖原图目录。

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `config.py` | CLIP 模型、图片尺寸、batch size 和输出路径配置。 |
| `dataset.py` | 读取图片 manifest 和 WebP 缩略图。 |
| `encoder.py` | 标准 CLIP 图像编码器。 |
| `encoder_local.py` | 本地/离线兼容图像编码器。 |
| `pooling.py` | 同一天多张图片的池化逻辑。 |
| `run_image_pipeline.py` | 图片编码入口。 |

## 输入

```text
1_data_handling/raw/image/image_manifest_commons_cleaned.jsonl.gz
1_data_handling/raw/image/thumbnails_webp/
2_encoding_feature/outputs/daily/structured_features/structured_index.npy
```

图片 manifest 覆盖 `1461/1461` 天，来源包括新闻图、NASA GIBS、Copernicus OGC、Wikimedia、Unsplash 和 Pexels。

## 输出

```text
2_encoding_feature/outputs/daily/image_features/
├─ image_embeddings.npy
├─ image_features_daily.csv
├─ image_counts.npy
├─ image_missing_flags.npy
├─ image_index.npy
└─ image_months.npy
```

`image_missing_flags.npy` 用于确认日频图片覆盖；当前主线目标是缺失总和为 0。

## 可调整参数

| 参数 | 影响 |
| --- | --- |
| `CLIP_MODEL_NAME` | 更换图像语义模型。 |
| `IMAGE_SIZE` | 控制 CLIP 输入尺寸。 |
| `BATCH_SIZE` | 控制吞吐和显存。 |
| pooling 策略 | 控制同一天多张图片如何汇总。 |

## 运行

```powershell
python 2_encoding_feature/image_clip/run_image_pipeline.py --frequency daily
```

重跑图片编码后，需要继续重跑 fusion 和 time-series，保证建模窗口读取最新图片向量。
