# baselines 交付说明

本目录保留基线模型源码和对应正式结果说明，不包含本地训练缓存。

## 交付内容

- HAR 与 LSTM 基线运行脚本
- 与基线结果复核有关的公共代码

## 本地重建或附加内容

- 基线的中间训练缓存和本地检查点不纳入交付目录
- 正式指标表和正式图统一保存在 `3_modeling/results/official/`

## 本地重建入口

```powershell
python 3_modeling/run_modeling_benchmarks.py
```
