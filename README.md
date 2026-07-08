# DSP AB 实验分析管道

> A step-by-step AB experiment analysis pipeline for DSP advertiser-side metrics. Retro-AA health checks → SRM validation → significance testing → chain-rule decomposition. Built with Claude Code.

面向 DSP 广告主侧的 AB 实验全链路分析工具。实现了 Retro-AA 健康检查、SRM 样本比例校验、显著性检验、链式归因分解——覆盖实验平台从"能不能信"到"为什么变"的完整方法论。

## 架构

```
           ┌──────────┐
           │ 原始 CSV  │
           └────┬─────┘
                ▼
     ┌──────────────────┐
     │ 01 数据清洗       │  缺失检测 / IQR 异常值 / 逻辑校验
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │ 02 Retro-AA      │  历史数据 1000 次随机切分 → 假阳性率检查
     │   健康检查        │  FPR ≤ 6% PASS / 6~8% WARN / >8% BLOCK
     └────┬──┬──┬───────┘
          │  │  │
      PASS WARN FAIL ← 阻断流程，排查分桶/埋点/数据管道
          │  │
          ▼  ▼
     ┌──────────────────┐
     │ 03 AB 分桶 +     │  分层随机分配 + 卡方 SRM 校验
     │    SRM 验证      │  维度均衡 + SMD < 0.1
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │ 04 显著性检验     │  Z-test / Welch T / Mann-Whitney U
     │                  │  Bonferroni + Benjamini-Hochberg FDR
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │ 归因分解（根因）   │  乘法型：链式分解 (Revenue = 曝光×CTR×CVR×ARPConv)
     │                  │  除法型：差分分解 (ROAS = Revenue/Spend)
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │ 06 告警引擎       │  流量偏移 / 归因异常 / 指标退化 / 离群值
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │ 05 Markdown 看板  │  健康检查 → AB 指标 → 归因分解 → 下钻 → 告警
     └──────────────────┘
```

## 快速开始

```bash
pip install -r requirements.txt
```

### 交互模式（逐步执行，每步确认）

```bash
python run_workflow.py
```

### 自动模式

```bash
python run_workflow.py --auto
```

### 单独运行某一步

```bash
python scripts/01_data_cleaner.py --input data/raw/xxx.csv --output-dir data/output
python scripts/02_retro_aa.py --input data/output/cleaned_data.csv --iterations 1000
python scripts/03_experiment_splitter.py --input data/output/cleaned_data.csv --mode simulate
python scripts/04_significance_test.py --input data/output/experiment_data.csv
python scripts/06_alerting.py --input data/output/experiment_data.csv
```

### 真实实验数据模式

```bash
python run_workflow.py --mode real --group-col experiment_group
```

## 项目结构

```
├── run_workflow.py              # 交互式管道入口
├── requirements.txt             # numpy, scipy, pandas, pyyaml
├── config/
│   └── experiment.yaml          # 实验参数：α 值、AA 迭代次数、告警阈值等
├── data/
│   ├── raw/                     # 原始 CSV 数据
│   └── output/                  # 中间文件（gitignored）
├── scripts/
│   ├── 01_data_cleaner.py       # 数据清洗 & 质量报告
│   ├── 02_retro_aa.py           # Retro-AA 健康检查
│   ├── 03_experiment_splitter.py# AB 分桶 & SRM 校验
│   ├── 04_significance_test.py  # 显著性检验 & 多重比较校正
│   ├── 05_dashboard.py         # Markdown 看板生成
│   └── 06_alerting.py          # 告警规则引擎
├── utils/
│   ├── metrics.py               # 广告主侧指标计算
│   ├── decomposition.py         # 链式分解 & 差分分解
│   └── visualize.py             # ASCII 图表 & 文本可视化
├── reports/                     # 生成报告
└── .claude/commands/
    └── dsp-ab.md                # Claude Code Skill
```

## 方法论

### AA → AB 框架

做 AB 之前先做 AA——先确认尺子没歪，再讨论量出来的长度。

| 检查 | 方法 | 判定标准 | 不通过则 |
|------|------|---------|---------|
| Retro-AA | 1000 次随机切分，统计 FPR | FPR ≤ 6% PASS / > 8% FAIL | 排查分桶 key / 埋点口径 / 数据管道 |
| SRM | 卡方检验样本比例 | p > 0.05 | 检查分桶代码 / 日志丢失 |
| 分层均衡 | 各维度卡方检验 | p > 0.05 | 检查分层抽样逻辑 |
| SMD | 标准化均值差异 | < 0.1 | 检查预实验指标基线 |

### 归因分解

显著性告诉你*哪个*指标变了，归因分解告诉你*为什么*变。

| 指标类型 | 方法 | 公式 |
|----------|------|------|
| 乘法型 | 链式分解 (Chain Rule) | Revenue = Impressions × CTR × CVR × ARPConv |
| 除法型 | 差分分解 (Difference Decomp) | ΔR ≈ ΔN/D₀ − N₀×ΔD/D₀² |

两层归因策略：先对 ROAS 做差分分解定位收入侧还是成本侧 → 再对问题侧做链式分解定位到具体因子。

### 显著性检验

| 指标 | 检验方法 |
|------|----------|
| CTR, CVR | Two-proportion Z-test |
| ROAS, CPC, CPM, CPA | Welch's T-test |
| Revenue, Spend | Mann-Whitney U |
| 多重比较校正 | Bonferroni + Benjamini-Hochberg (FDR) |

## 视角说明

这是 **DSP 广告主侧** 分析，关注广告主花了多少钱、拿到多少效果。不是媒体变现侧（eCPM、填充率是媒体平台/App 开发者看的指标）。

## 数据字段

| 字段 | 含义 | 类型 |
|------|------|------|
| date | 日期 | 维度 |
| platform | 投放媒体 | 维度 |
| campaign_type | 素材类型 | 维度 |
| industry | 广告主行业 | 维度 |
| country | 目标国家 | 维度 |
| impressions | 曝光量 | 基础指标 |
| clicks | 点击量 | 基础指标 |
| ad_spend | 消耗 | 基础指标 |
| conversions | 转化量 | 基础指标 |
| revenue | 转化营收 | 基础指标 |
