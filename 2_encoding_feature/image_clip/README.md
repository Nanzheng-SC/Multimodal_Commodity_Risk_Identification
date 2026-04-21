# image_clip 交付说明

本目录保留图像编码源码，不包含图像 embedding 数组。

## 交付内容

- 图像数据集定义、编码器、池化逻辑和运行脚本

## 本地重建或附加内容

- 图像编码结果写入 `2_encoding_feature/outputs/`
- 本地下载的模型缓存不纳入交付目录

## 本地重建入口

```powershell
python 2_encoding_feature/image_clip/run_image_pipeline.py --frequency daily
```
