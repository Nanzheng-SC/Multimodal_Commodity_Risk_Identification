# baselines GitHub 上传说明

本目录上传基线模型源码和对应正式结果说明，不上传本地训练缓存。

## 上传保留

- HAR 与 LSTM 基线运行脚本
- 与基线结果复核有关的公共代码

## 本地生成或忽略

- 基线的中间训练缓存和本地检查点不上传
- 正式指标表和正式图统一保存在 `3_modeling/results/official/`

## 本地重建入口

```powershell
python 3_modeling/run_modeling_benchmarks.py
```
