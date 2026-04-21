# image_clip GitHub 上传说明

本目录上传图像编码源码，不上传图像 embedding 数组。

## 上传保留

- 图像数据集定义、编码器、池化逻辑和运行脚本

## 本地生成或忽略

- 图像编码结果写入 `2_encoding_feature/outputs/`，不上传
- 本地下载的模型缓存不上传

## 本地重建入口

```powershell
python 2_encoding_feature/image_clip/run_image_pipeline.py --frequency daily
```
