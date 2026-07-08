"""
脚本 03：AB 实验分桶 + SRM 均衡校验
====================================
支持两种模式：
A. 模拟模式：按分层维度随机抽样，分为 control/treatment
B. 真实模式：读取用户提供的 experiment_group 列

均衡校验（SRM + 分层均衡 + 预实验指标校验）：
- SRM: 卡方检验验证 50/50
- 分层均衡: 每个维度做卡方检验
- SMD: Standardized Mean Difference < 0.1
- 每日流量偏移检查
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

from utils.metrics import calculate_all_metrics, aggregate_metrics
from utils.visualize import format_smd, format_pvalue


def simulate_split(
    df: pd.DataFrame,
    strata_cols: list,
    split_ratio: float = 0.5,
    random_seed: int = 42,
    control_label: str = 'control',
    treatment_label: str = 'treatment',
) -> pd.DataFrame:
    """
    分层随机抽样模拟 AB 分桶。
    按 strata_cols 分层，每层内随机分配 control/treatment。
    """
    np.random.seed(random_seed)
    df = df.copy()

    # 创建分层 key
    df['_strata_key'] = ''
    for col in strata_cols:
        if col in df.columns:
            df['_strata_key'] += df[col].astype(str) + '|'

    df['group'] = None

    for key, group_df in df.groupby('_strata_key'):
        n = len(group_df)
        indices = group_df.index.values
        np.random.shuffle(indices)
        # 用二项分布随机分配，避免 int() 向下取整造成的系统性偏差
        n_ctrl = np.random.binomial(n, split_ratio)
        n_ctrl = max(1, min(n_ctrl, n - 1)) if n > 1 else (1 if np.random.random() < 0.5 else 0)
        df.loc[indices[:n_ctrl], 'group'] = control_label
        df.loc[indices[n_ctrl:], 'group'] = treatment_label

    df.drop(columns=['_strata_key'], inplace=True)

    n_ctrl = (df['group'] == control_label).sum()
    n_treat = (df['group'] == treatment_label).sum()
    print(f"\n[03_splitter] 分层随机分桶完成:")
    print(f"  Control: {n_ctrl} 行 ({n_ctrl/len(df)*100:.1f}%)")
    print(f"  Treatment: {n_treat} 行 ({n_treat/len(df)*100:.1f}%)")
    print(f"  分层维度: {', '.join(strata_cols)}")

    return df


def real_split(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """读取已有的实验分组列"""
    if group_col not in df.columns:
        raise ValueError(f"分组列 '{group_col}' 不存在。可用列: {list(df.columns)}")

    df = df.copy()
    df['group'] = df[group_col]
    groups = df['group'].value_counts()
    print(f"\n[03_splitter] 读取已有实验分组:")
    for g, n in groups.items():
        print(f"  {g}: {n} 行 ({n/len(df)*100:.1f}%)")

    return df


def srm_test(df: pd.DataFrame, expected_ratio: float = 0.5, alpha: float = 0.05) -> dict:
    """
    SRM (Sample Ratio Mismatch) 检验。
    卡方检验：实际样本比例是否偏离预期 50/50。
    """
    counts = df['group'].value_counts()
    expected_total = len(df)
    expected_counts = {
        'control': expected_total * expected_ratio,
        'treatment': expected_total * (1 - expected_ratio),
    }

    observed = []
    expected = []
    labels = []
    for label in expected_counts:
        if label in counts.index:
            observed.append(counts[label])
            expected.append(expected_counts[label])
            labels.append(label)

    if len(observed) < 2:
        return {'passed': False, 'p_value': 1.0, 'reason': '只有一个分组'}

    chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)

    control_pct = observed[0] / sum(observed) * 100 if sum(observed) > 0 else 0

    return {
        'passed': p >= alpha,
        'chi2': round(chi2, 4),
        'p_value': round(p, 4),
        'observed_ratio': f"{observed[0]}/{observed[1]}" if len(observed) == 2 else str(observed),
        'control_pct': round(control_pct, 1),
        'expected_ratio': f"{expected_ratio*100:.0f}/{100-expected_ratio*100:.0f}",
        'alpha': alpha,
    }


def strata_balance_test(df: pd.DataFrame, strata_cols: list, alpha: float = 0.05) -> list:
    """
    分层均衡检验：对每个维度做卡方检验，
    验证两组在各维度上的分布是否一致。
    """
    results = []
    for col in strata_cols:
        if col not in df.columns:
            continue

        # 交叉表：维度类别 × 分组
        ctab = pd.crosstab(df[col], df['group'])

        if ctab.shape[1] < 2:
            results.append({
                'dimension': col,
                'passed': False,
                'p_value': 1.0,
                'reason': '只有一个分组',
            })
            continue

        chi2, p, dof, expected = stats.chi2_contingency(ctab)

        results.append({
            'dimension': col,
            'passed': p >= alpha,
            'chi2': round(chi2, 4),
            'p_value': round(p, 4),
            'dof': dof,
            'categories': list(ctab.index),
            'observations': ctab.to_dict(),
        })

    return results


def smd_check(df: pd.DataFrame, baseline_metrics: list, alpha: float = 0.05) -> list:
    """
    SMD (Standardized Mean Difference) 检验。
    比较两组在预实验指标上的标准化均值差异。
    SMD < 0.1 → 均衡。
    """
    results = []
    control = df[df['group'] == 'control']
    treatment = df[df['group'] == 'treatment']

    for metric in baseline_metrics:
        if metric not in df.columns:
            continue

        c_vals = control[metric].dropna()
        t_vals = treatment[metric].dropna()

        mean_c = c_vals.mean()
        mean_t = t_vals.mean()
        var_c = c_vals.var()
        var_t = t_vals.var()

        # Pooled SD
        n_c, n_t = len(c_vals), len(t_vals)
        pooled_sd = np.sqrt(((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2))
        smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0

        # T 检验 p-value
        try:
            _, t_p = stats.ttest_ind(c_vals, t_vals, equal_var=False)
        except Exception:
            t_p = 1.0

        results.append({
            'metric': metric,
            'control_mean': round(mean_c, 4),
            'treatment_mean': round(mean_t, 4),
            'smd': round(smd, 4),
            'smd_status': '✓ 均衡' if abs(smd) < 0.1 else '⚠ 偏差',
            't_test_pvalue': round(t_p, 4),
            't_test_passed': t_p >= alpha,
        })

    return results


def daily_traffic_check(df: pd.DataFrame, threshold: float = 0.55) -> list:
    """每日流量偏移检查"""
    alerts = []
    for date, day_df in df.groupby('date'):
        counts = day_df['group'].value_counts()
        total = len(day_df)
        if total == 0:
            continue
        ctrl_pct = counts.get('control', 0) / total
        if ctrl_pct > threshold or ctrl_pct < (1 - threshold):
            alerts.append({
                'date': str(date.date() if hasattr(date, 'date') else date),
                'total': total,
                'control_pct': round(ctrl_pct * 100, 1),
            })
    return alerts


def generate_split_report(
    srm: dict,
    strata_results: list,
    smd_results: list,
    traffic_alerts: list,
    output_dir: str,
) -> str:
    """生成分桶均衡报告"""
    lines = []
    lines.append("# AB 分桶 & 均衡校验报告\n")

    # SRM
    lines.append("## SRM 样本比例检查\n")
    if srm['passed']:
        lines.append(f"✓ **SRM 通过** (p={srm['p_value']}), 实际比例={srm['observed_ratio']}")
    else:
        lines.append(f"✗ **SRM 不通过** (p={srm['p_value']}), 实际比例={srm['observed_ratio']}")
        lines.append(f"  ⚠ 样本比例显著偏离预期 {srm['expected_ratio']}，可能存在分桶 bug！")

    # 分层均衡
    lines.append("\n## 分层维度均衡检验\n")
    lines.append("| 维度 | 卡方 p-value | 结论 |")
    lines.append("|------|-------------|------|")
    for r in strata_results:
        status = "✓ 均衡" if r['passed'] else "✗ 不均衡"
        lines.append(f"| {r['dimension']} | {r['p_value']} | {status} |")

    # SMD
    lines.append("\n## 预实验指标均衡 (SMD)\n")
    lines.append("| 指标 | Control均值 | Treatment均值 | SMD | 结论 | T-test p |")
    lines.append("|------|------------|--------------|-----|------|---------|")
    for r in smd_results:
        lines.append(
            f"| {r['metric']} | {r['control_mean']} | {r['treatment_mean']} | "
            f"{r['smd']} | {r['smd_status']} | {r['t_test_pvalue']} |"
        )

    # 每日偏移
    if traffic_alerts:
        lines.append("\n## ⚠ 每日流量偏移告警\n")
        for a in traffic_alerts:
            lines.append(f"- {a['date']}: Control 占比 {a['control_pct']}% (共 {a['total']} 条)")

    report_path = os.path.join(output_dir, 'split_balance_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n[03_splitter] 均衡报告已保存: {report_path}")

    return '\n'.join(lines)


def main(
    df: pd.DataFrame,
    config: dict,
    output_dir: str,
) -> tuple:
    """主流程"""
    split_config = config.get('splitting', {})
    mode = split_config.get('mode', 'simulate')
    group_col = split_config.get('group_col', 'group')
    strata_cols = split_config.get('strata_dimensions', ['platform', 'campaign_type', 'industry', 'country'])
    split_ratio = split_config.get('split_ratio', 0.5)
    seed = split_config.get('random_seed', 42)
    balance_config = config.get('balance_check', {})
    balance_alpha = balance_config.get('alpha', 0.05)
    baseline_metrics = balance_config.get('baseline_metrics', ['impressions', 'clicks', 'conversions', 'spend', 'revenue'])

    # 1. 分桶
    if mode == 'real':
        df = real_split(df, group_col)
    else:
        df = simulate_split(df, strata_cols, split_ratio, seed)

    # 2. SRM
    srm = srm_test(df, split_ratio, balance_alpha)

    # 3. 分层均衡
    strata_results = strata_balance_test(df, strata_cols, balance_alpha)

    # 4. SMD
    smd_results = smd_check(df, baseline_metrics, balance_alpha)

    # 5. 每日流量检查
    traffic_alerts = daily_traffic_check(df, config.get('alerting', {}).get('traffic_skew_ratio', 0.55))

    # 6. 报告
    report = generate_split_report(srm, strata_results, smd_results, traffic_alerts, output_dir)

    # 7. 总判定
    all_passed = srm['passed'] and all(r['passed'] for r in strata_results)
    smd_all_ok = all(abs(r['smd']) < 0.1 for r in smd_results)

    print(f"\n[03_splitter] {'='*50}")
    print(f"[03_splitter] SRM: {'✓' if srm['passed'] else '✗'} | "
          f"分层均衡: {'✓' if all(r['passed'] for r in strata_results) else '✗'} | "
          f"SMD: {'✓' if smd_all_ok else '⚠'}")

    # 保存分桶数据
    output_path = os.path.join(output_dir, 'experiment_data.csv')
    df.to_csv(output_path, index=False)
    print(f"[03_splitter] 实验数据已保存: {output_path}")

    return df, srm, strata_results, smd_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/output/cleaned_data.csv')
    parser.add_argument('--mode', default='simulate', choices=['simulate', 'real'])
    parser.add_argument('--group-col', default='group')
    parser.add_argument('--output-dir', default='data/output')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    import yaml
    try:
        with open('config/experiment.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}
    config.setdefault('splitting', {})['mode'] = args.mode
    config.setdefault('splitting', {})['group_col'] = args.group_col
    df, srm, strata, smd = main(df, config, args.output_dir)
