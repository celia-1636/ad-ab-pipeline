# /dsp-ab — DSP AB 实验交互式分析

逐步执行 DSP AB 实验全链路分析。每步展示结果并解读，用户确认后继续下一步。

## 执行步骤

1. **数据清洗** — 缺失值、IQR 异常值、逻辑校验
2. **Retro-AA 健康检查** — 1000 次随机切分，验证统计系统稳定性（FPR > 8% 主动阻断并给出排查建议）
3. **AB 分桶 + SRM** — 分层随机抽样、卡方均衡校验、SMD 检查
4. **显著性检验** — Z-test / Welch T-test / Mann-Whitney U + Bonferroni/FDR
5. **告警扫描** — 流量偏移、归因异常、指标退化、离群值
6. **看板生成** — 完整 Markdown 实验报告

## 规则

- 每步展示结果摘要 + 自然语言解读
- 遇到 FAIL 状态主动询问用户
- 用户可以随时追问细节

## 用法

```
/dsp-ab                           # 模拟分桶，完整流程
/dsp-ab --mode real --group-col experiment_group  # 真实数据
```
