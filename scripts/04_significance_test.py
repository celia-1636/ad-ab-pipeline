"""
脚本 04：显著性检验
===================
对 AB 实验的 Control vs Treatment 两组指标进行显著性检验，
含多重比较校正、效应量计算、维度下钻。

检验方法：
- CTR, CVR → Two-proportion Z-test
- ROAS, CPC, CPM, CPA → Welch's T-test
- Revenue, Spend → Mann-Whitney U (非参数)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from itertools import combinations

from utils.metrics import calculate_all_metrics, aggregate_metrics
from utils.visualize import format_pvalue, format_pct_change


def two_proportion_ztest(
    n1: int, x1: float,
    n2: int, x2: float,
    alternative: str = 'two-sided',
) -> Tuple[float, float, float]:
    """
    双样本比例 Z 检验。
    n1, n2: 两组样本量（对于 CTR，n=impressions；对于 CVR，n=clicks）
    x1, x2: 两组成功次数（对于 CTR，x=clicks；对于 CVR，x=conversions）
    """
    p1 = x1 / n1 if n1 > 0 else 0
    p2 = x2 / n2 if n2 > 0 else 0
    p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 0.0, p1, p2

    z = (p1 - p2) / se

    if alternative == 'two-sided':
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == 'greater':
        p = 1 - stats.norm.cdf(z)
    else:
        p = stats.norm.cdf(z)

    return p, p1, p2


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h：比例指标的效应量"""
    p1 = np.clip(p1, 0.0001, 0.9999)
    p2 = np.clip(p2, 0.0001, 0.9999)
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Cohen's d：连续指标的效应量"""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = x1.var(ddof=1)
    var2 = x2.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (x1.mean() - x2.mean()) / pooled_sd


def interpret_effect_size(d: float) -> str:
    """解读效应量大小"""
    ad = abs(d)
    if ad < 0.2:
        return "微小"
    elif ad < 0.5:
        return "小"
    elif ad < 0.8:
        return "中等"
    else:
        return "大"


def test_metric(
    df: pd.DataFrame,
    metric: str,
    alpha: float = 0.05,
    alternative: str = 'two-sided',
) -> Dict:
    """
    对单个指标进行 Control vs Treatment 显著性检验。
    自动选择检验方法。
    """
    control = df[df['group'] == 'control']
    treatment = df[df['group'] == 'treatment']

    # --- 比例指标（按聚合值算）---
    if metric == 'CTR':
        # CTR = clicks / impressions（用聚合后的总点击/总曝光）
        ctrl_p, ctrl_n = control['clicks'].sum(), control['impressions'].sum()
        treat_p, treat_n = treatment['clicks'].sum(), treatment['impressions'].sum()
        p, v1, v2 = two_proportion_ztest(treat_n, treat_p, ctrl_n, ctrl_p, alternative)
        effect = cohens_h(v1, v2)
        control_val, treatment_val = v2, v1

    elif metric == 'CVR':
        # CVR = conversions / clicks
        ctrl_p, ctrl_n = control['conversions'].sum(), control['clicks'].sum()
        treat_p, treat_n = treatment['conversions'].sum(), treatment['clicks'].sum()
        p, v1, v2 = two_proportion_ztest(treat_n, treat_p, ctrl_n, ctrl_p, alternative)
        effect = cohens_h(v1, v2)
        control_val, treatment_val = v2, v1

    # --- 连续指标 ---
    elif metric in ['ROAS', 'CPC', 'CPA', 'CPM']:
        ctrl_vals = control[metric].replace([np.inf, -np.inf], np.nan).dropna()
        treat_vals = treatment[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ctrl_vals) < 2 or len(treat_vals) < 2:
            p = 1.0
            effect = 0
        else:
            t_stat, p = stats.ttest_ind(treat_vals, ctrl_vals, equal_var=False, alternative=alternative)
            effect = cohens_d(treat_vals, ctrl_vals)
        control_val = ctrl_vals.mean()
        treatment_val = treat_vals.mean()

    # --- 非参数指标 ---
    elif metric in ['revenue', 'spend', 'Revenue', 'Spend']:
        ctrl_vals = control[metric].dropna()
        treat_vals = treatment[metric].dropna()
        try:
            u_stat, p = stats.mannwhitneyu(treat_vals, ctrl_vals, alternative=alternative)
            effect = cohens_d(treat_vals, ctrl_vals)
        except Exception:
            p = 1.0
            effect = 0
        control_val = ctrl_vals.mean()
        treatment_val = treat_vals.mean()

    else:
        ctrl_vals = control[metric].dropna()
        treat_vals = treatment[metric].dropna()
        try:
            t_stat, p = stats.ttest_ind(treat_vals, ctrl_vals, equal_var=False)
            effect = cohens_d(treat_vals, ctrl_vals)
        except Exception:
            p = 1.0
            effect = 0
        control_val = ctrl_vals.mean()
        treatment_val = treat_vals.mean()

    # 计算 lift%
    lift_pct = (treatment_val - control_val) / abs(control_val) * 100 if control_val != 0 else 0

    return {
        'metric': metric,
        'control_value': control_val,
        'treatment_value': treatment_val,
        'delta': treatment_val - control_val,
        'lift_pct': lift_pct,
        'p_value': p,
        'significant': p < alpha,
        'effect_size': effect,
        'effect_magnitude': interpret_effect_size(effect),
        'test_method': ('proportion_z' if metric in ['CTR', 'CVR']
                        else 'mannwhitney' if metric.lower() in ['revenue', 'spend']
                        else 'welch_t'),
    }


def multiple_comparison_correction(results: List[Dict], alpha: float = 0.05) -> List[Dict]:
    """
    多重比较校正：Bonferroni + Benjamini-Hochberg (FDR)。
    对一组检验的 p-value 进行校正，避免多重比较导致假阳性膨胀。
    """
    if not results:
        return results

    n = len(results)
    pvalues = np.array([r['p_value'] for r in results])

    # Bonferroni
    bonferroni_alpha = alpha / n

    # Benjamini-Hochberg FDR
    ranks = np.argsort(np.argsort(pvalues)) + 1  # rank from 1
    bh_thresholds = ranks / n * alpha

    for i, r in enumerate(results):
        r['bonferroni_alpha'] = bonferroni_alpha
        r['bonferroni_significant'] = r['p_value'] < bonferroni_alpha
        r['fdr_threshold'] = bh_thresholds[i]
        r['fdr_significant'] = r['p_value'] < bh_thresholds[i]

    return results


def drilldown_analysis(
    df: pd.DataFrame,
    dimension: str,
    metrics: list,
    alpha: float = 0.05,
) -> Dict[str, List[Dict]]:
    """
    维度下钻：按指定维度分别做显著性检验。
    例如：按 platform 维度，对每个平台分别检验 Control vs Treatment。
    """
    results = {}
    for category, cat_df in df.groupby(dimension):
        cat_results = []
        for metric in metrics:
            try:
                r = test_metric(cat_df, metric, alpha)
                cat_results.append(r)
            except Exception as e:
                cat_results.append({
                    'metric': metric,
                    'error': str(e),
                    'p_value': 1.0,
                    'significant': False,
                })
        results[category] = cat_results

    return results


def generate_significance_report(
    results: List[Dict],
    drilldown: Dict[str, Dict] = None,
    output_dir: str = None,
) -> str:
    """生成显著性检验报告"""
    lines = []
    lines.append("# AB 实验显著性检验报告\n")

    # 主指标表
    lines.append("## 核心指标对比\n")
    lines.append("| 指标 | Control | Treatment | Lift% | P-value | 显著性 | Effect Size |")
    lines.append("|------|---------|-----------|-------|---------|--------|-------------|")
    for r in results:
        sig_str = "✓" if r.get('significant') else ""
        fdr_sig = " ✓(FDR)" if r.get('fdr_significant') else ""
        lines.append(
            f"| {r['metric']} | {r['control_value']:.4f} | {r['treatment_value']:.4f} | "
            f"{r['lift_pct']:+.2f}% | {format_pvalue(r['p_value'])} | "
            f"{sig_str}{fdr_sig} | {r['effect_size']:.4f} ({r['effect_magnitude']}) |"
        )

    # 多重比较校正说明
    lines.append(f"\n> 多重比较校正: Bonferroni + Benjamini-Hochberg (FDR)")
    lines.append(f"> 显著性标记: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1, ns not significant")

    # 维度下钻
    if drilldown:
        lines.append("\n## 维度下钻分析\n")
        for dim, cat_results in drilldown.items():
            # cat_results is {category_name: [result_list]}
            lines.append(f"\n### {dim}\n")
            lines.append("| 维度值 | 显著指标 | 详情 |")
            lines.append("|--------|---------|------|")
            for cat, res_list in cat_results.items():
                sig_items = [r for r in res_list if r.get('significant')]
                if sig_items:
                    details = ', '.join(
                        f"{r['metric']}({r['lift_pct']:+.1f}%)" for r in sig_items
                    )
                    lines.append(f"| {cat} | {len(sig_items)} 个 | {details} |")
                else:
                    lines.append(f"| {cat} | 0 个 | 无显著差异 |")
            lines.append("")

    # 保存
    if output_dir:
        report_path = os.path.join(output_dir, 'significance_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"[04_significance] 报告已保存: {report_path}")

    return '\n'.join(lines)


def main(
    df: pd.DataFrame,
    config: dict,
    output_dir: str,
) -> Tuple[List[Dict], dict]:
    """主流程"""
    sig_config = config.get('significance', {})
    alpha = sig_config.get('alpha', 0.05)
    alternative = sig_config.get('alternative', 'two-sided')
    corrections = sig_config.get('multiple_comparison_correction', ['bonferroni', 'fdr_bh'])

    # 确保指标列存在
    df = calculate_all_metrics(df)

    # 核心指标
    core_metrics = ['CTR', 'CVR', 'ROAS', 'CPC', 'CPA', 'CPM', 'revenue']

    print(f"[04_significance] 检验 {len(core_metrics)} 个核心指标 (α={alpha})")

    # 1. 主检验
    results = []
    for metric in core_metrics:
        try:
            r = test_metric(df, metric, alpha, alternative)
            results.append(r)
        except Exception as e:
            print(f"  ⚠ {metric} 检验失败: {e}")
            results.append({'metric': metric, 'error': str(e), 'significant': False})

    # 2. 多重比较校正
    results = multiple_comparison_correction(results, alpha)

    # 3. 维度下钻
    dimensions = ['platform', 'campaign_type', 'industry']
    drilldown = {}
    for dim in dimensions:
        if dim in df.columns:
            drilldown[dim] = drilldown_analysis(df, dim, ['CTR', 'CVR', 'ROAS'], alpha)

    # 4. 打印摘要
    sig_count = sum(1 for r in results if r.get('significant'))
    fdr_sig_count = sum(1 for r in results if r.get('fdr_significant'))
    print(f"[04_significance] 显著指标: {sig_count}/{len(core_metrics)} (原始), "
          f"{fdr_sig_count}/{len(core_metrics)} (FDR 校正后)")

    for r in results:
        if r.get('significant'):
            print(f"  ★ {r['metric']}: {r['control_value']:.4f} → {r['treatment_value']:.4f} "
                  f"({r['lift_pct']:+.2f}%, p={r['p_value']:.4f}, d={r['effect_size']:.3f})")

    # 5. 生成报告
    report = generate_significance_report(results, drilldown, output_dir)

    return results, drilldown


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/output/experiment_data.csv')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--output-dir', default='data/output')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    config = {'significance': {'alpha': args.alpha}}
    main(df, config, args.output_dir)
