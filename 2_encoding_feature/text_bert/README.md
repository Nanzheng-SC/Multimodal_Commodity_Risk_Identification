# text_bert GitHub 上传说明

本目录上传文本编码源码和轻量分词器文件，不上传大体积模型权重和编码结果。

## 上传保留

- 文本数据集定义、编码器、池化逻辑和运行脚本
- `models/bert-base-multilingual-cased/` 下的轻量配置文件、`tokenizer.json`、`vocab.txt`

## 本地生成或忽略

- `model.safetensors`、`pytorch_model.bin` 等大体积权重不上传
- 文本 embedding 数组写入 `2_encoding_feature/outputs/`，不上传

## 本地重建入口

```powershell
python 2_encoding_feature/text_bert/download_multilingual_model.py
python 2_encoding_feature/text_bert/run_text_pipeline.py --frequency daily
```
