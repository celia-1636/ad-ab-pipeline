"""
脚本 02：Retro-AA 健康检查
==========================
用历史数据反复随机切分 N 次（默认 1000 次），
每次分成两组跑同样的显著性检验，
验证"无差异时系统会不会制造差异"。

核心概念（来源：小红书 AA 实验文章 + 大厂实践）：
- Retro-AA 不是更高级的 AB，Retro-AA 是 AB 之前的体检
- 检查 p-value 分布是否均匀，假阳性率是否接近名义 α
- FPR ≤ 6% → PASS / 6-8% → WARN / >8% → FAIL（阻断）

三种健康检查：
1. Online A/A: 线上真实分流，端到端验证（本项目用 CSV 无法模拟）
2. Retro-AA: 历史数据反复切分，验证指标/统计系统稳定性 ★
3. SRM: 实验运行中检查样本比例（见 03_experiment_splitter.py）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime
from collections import defaultdict

from utils.metrics import calculate_all_metrics
from utils.visualize import fpr_status, pvalue_distribution_histogram, ascii_bar_chart


def random_split(df: pd.DataFrame, seed: int = None) -> tuple:
    """随机 50/50 切分数据"""
    n = len(df)
    indices = np.random.RandomState(seed).permutation(n)
    half = n // 2
    return df.iloc[indices[:half]], df.iloc[indices[half:]]


def test_proportion(group_a: pd.Series, group_b: pd.Series) -> float:
    """两组比例指标的双样本 Z 检验，返回 p-value"""
    n_a, n_b = len(group_a), len(group_b)
    if n_a == 0 or n_b == 0:
        return 1.0

    # 聚合求和来做比例检验
    x_a = group_a.sum()
    x_b = group_b.sum()

    p_pool = (x_a + x_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    if se == 0:
        return 1.0

    z = (x_a/n_a - x_b/n_b) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return p


def test_continuous(group_a: pd.Series, group_b: pd.Series) -> float:
    """两组连续指标的 Welch T 检验，返回 p-value"""
    a = group_a.dropna()
    b = group_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return 1.0
    try:
        t_stat, p = stats.ttest_ind(a, b, equal_var=False)
        return p
    except Exception:
        return 1.0


def run_retro_aa(df: pd.DataFrame, metrics: list, iterations: int = 1000, seed: int = 42) -> dict:
    """
    执行 Retro-AA：N 次随机切分 → 每次对每个指标跑显著性检验 → 统计假阳性率
    """
    print(f"[02_retro_aa] 开始 Retro-AA 健康检查: {iterations} 次迭代, {len(metrics)} 个指标")
    print(f"[02_retro_aa] 核心问题: 在没有策略差异时，统计系统会不会报出'显著'？")

    results = {m: [] for m in metrics}

    for i in range(iterations):
        g1, g2 = random_split(df, seed=seed + i)

        for metric in metrics:
            if metric in ['CTR', 'CVR']:
                # 比例指标：用 Z-test，需要聚合为每条数据的比例
                g1_vals = g1[metric].dropna()
                g2_vals = g2[metric].dropna()
                if len(g1_vals) > 0 and len(g2_vals) > 0:
                    # 对比例指标，我们用每组每条记录的原始比例做 t 检验
                    # （每条 row 本身就是一个比例值）
                    try:
                        _, p = stats.ttest_ind(g1_vals, g2_vals, equal_var=False)
                        results[metric].append(p)
                    except Exception:
                        results[metric].append(1.0)
                else:
                    results[metric].append(1.0)

            elif metric in ['CPC', 'CPA', 'ROAS', 'CPM']:
                # 连续指标：用 Welch T-test
                g1_vals = g1[metric].replace([np.inf, -np.inf], np.nan).dropna()
                g2_vals = g2[metric].replace([np.inf, -np.inf], np.nan).dropna()
                try:
                    _, p = stats.ttest_ind(g1_vals, g2_vals, equal_var=False)
                    results[metric].append(p)
                except Exception:
                    results[metric].append(1.0)

        if (i + 1) % 250 == 0:
            print(f"  进度: {i+1}/{iterations}")

    return results


def analyze_retro_aa(results: dict, alpha: float = 0.05) -> dict:
    """分析 Retro-AA 结果"""
    analysis = {}
    all_pvalues = []

    for metric, pvalues in results.items():
        pvals = np.array(pvalues)
        n_sig = np.sum(pvals < alpha)
        fpr = n_sig / len(pvals)
        fpr_ci_lower = stats.binom.ppf(0.025, len(pvals), alpha) / len(pvals)
        fpr_ci_upper = stats.binom.ppf(0.975, len(pvals), alpha) / len(pvals)

        analysis[metric] = {
            'n_iterations': len(pvals),
            'n_significant': int(n_sig),
            'fpr': round(fpr, 4),
            'fpr_ci_lower': round(fpr_ci_lower, 4),
            'fpr_ci_upper': round(fpr_ci_upper, 4),
            'status': fpr_status(fpr),
            'mean_pvalue': round(float(np.mean(pvals)), 4),
            'median_pvalue': round(float(np.median(pvals)), 4),
        }
        all_pvalues.extend(pvals.tolist())

    # 整体 FPR
    overall_sig = np.sum(np.array(all_pvalues) < alpha)
    overall_fpr = overall_sig / len(all_pvalues)
    analysis['_overall'] = {
        'n_iterations_total': len(all_pvalues),
        'n_significant': int(overall_sig),
        'fpr': round(overall_fpr, 4),
        'status': fpr_status(overall_fpr),
    }

    return analysis


def generate_retro_aa_report(analysis: dict, results: dict, output_dir: str) -> str:
    """生成 Retro-AA 健康报告"""
    overall = analysis.pop('_overall')
    lines = []
    lines.append("# Retro-AA 健康检查报告\n")
    lines.append(f"## 整体结论: {overall['status']}\n")
    lines.append(f"- 总检验次数: {overall['n_iterations_total']}")
    lines.append(f"- 显著次数: {overall['n_significant']}")
    lines.append(f"- 假阳性率 (FPR): {overall['fpr']:.4f} (名义 α=0.05)")
    lines.append(f"- 判定: {overall['status']}")

    if overall['fpr'] > 0.08:
        lines.append("\n⚠️ **FPR 超过 8%，建议阻断 AB 流程，排查以下可能原因：**")
        lines.append("1. 分桶 key 选错（用户 ID/设备 ID/Cookie 边界）")
        lines.append("2. 埋点口径不一致（曝光触发时机、日志漏报）")
        lines.append("3. 数据管道过滤规则（如 bot detection 误杀活跃用户）")
        lines.append("4. 实验污染（跨组用户、多实验互相影响）")
    elif overall['fpr'] > 0.06:
        lines.append("\n⚠️ FPR 略偏高（6-8%），记录风险但继续 AB。建议关注后续 SRM 检查。")
    else:
        lines.append("\n✓ 统计系统健康，可以继续 AB 实验。")

    lines.append("\n## 各指标详情\n")
    lines.append("| 指标 | 检验次数 | 显著次数 | FPR | 状态 |")
    lines.append("|------|---------|---------|-----|------|")
    for metric, info in analysis.items():
        lines.append(
            f"| {metric} | {info['n_iterations']} | {info['n_significant']} | "
            f"{info['fpr']} | {info['status']} |"
        )

    # p-value 分布（选 ROAS 做示例）
    if 'ROAS' in results:
        lines.append("\n## p-value 分布 (ROAS)\n")
        lines.append("```")
        lines.append(pvalue_distribution_histogram(results['ROAS'], title="ROAS p-value Distribution"))
        lines.append("```")

    # 保存报告
    report_path = os.path.join(output_dir, 'retro_aa_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n[02_retro_aa] 报告已保存: {report_path}")

    # 保存 JSON 结果
    json_path = os.path.join(output_dir, 'retro_aa_results.json')
    analysis['_overall'] = overall  # 放回去
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    print(f"[02_retro_aa] JSON 结果已保存: {json_path}")

    return '\n'.join(lines)


def main(df: pd.DataFrame, config: dict, output_dir: str):
    """主流程"""
    retro_config = config.get('retro_aa', {})
    iterations = retro_config.get('iterations', 1000)
    seed = retro_config.get('random_seed', 42)
    pass_threshold = retro_config.get('fpr_pass_threshold', 0.06)
    warn_threshold = retro_config.get('fpr_warn_threshold', 0.08)

    alpha = config.get('significance', {}).get('alpha', 0.05)

    # 计算所有指标
    df = calculate_all_metrics(df)

    # 核心指标列表
    metrics = ['CTR', 'CVR', 'CPC', 'CPA', 'ROAS', 'CPM']

    # 执行 Retro-AA
    results = run_retro_aa(df, metrics, iterations, seed)

    # 分析
    analysis = analyze_retro_aa(results, alpha)

    # 生成报告
    report = generate_retro_aa_report(analysis, results, output_dir)

    # 门控判定
    overall_fpr = analysis['_overall']['fpr']
    if overall_fpr > warn_threshold:
        status = 'FAIL'
    elif overall_fpr > pass_threshold:
        status = 'WARN'
    else:
        status = 'PASS'

    print(f"\n[02_retro_aa] {'='*50}")
    print(f"[02_retro_aa] Retro-AA 结论: {status} (FPR={overall_fpr:.4f})")

    return status, analysis


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/output/cleaned_data.csv')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--output-dir', default='data/output')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    config = {
        'retro_aa': {'iterations': args.iterations, 'random_seed': 42},
        'significance': {'alpha': 0.05}
    }
    status, _ = main(df, config, args.output_dir)
    print(f"\n[02_retro_aa] 最终状态: {status}")
