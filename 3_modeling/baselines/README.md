# baselines

本目录保存主线对照基线。它们的作用不是替代多模态主线，而是提供统一口径下的经典时序参考，用来说明 `TimeMixer + fusion + late.gru_gate` 的增益来自模型与模态两方面。

## 目录结构

| 文件或目录 | 作用 |
| --- | --- |
| `__init__.py` | 包标记。 |
| `har/run_har_benchmark.py` | HAR-no-leak 残差基线。 |
| `lstm/run_lstm_benchmark.py` | LSTM-window 残差基线。 |

## 作用

- `HAR` 提供低复杂度、可解释的传统时间序列基准。
- `LSTM` 提供深度时序神经网络基准。
- 二者都使用与主线一致的 horizon30 残差任务和窗口数据，保证比较公平。

## 输出位置

基线 official 结果统一输出到：

```text
3_modeling/results/official/daily_horizon30/
```

其中：

- `har_no_leak_residual/` 对应 HAR
- `lstm_residual/` 对应 LSTM
- `naive_reference/` 对应 Naive reference

## 当前意义

基线层的价值在于提供稳定参照。当前项目中，主线 `TimeMixer + fusion + late.gru_gate` 已经在 fixed test 和 rolling 两个口径下整体领先这些基线，因此基线目录主要承担“证明增益真实存在”的作用。
