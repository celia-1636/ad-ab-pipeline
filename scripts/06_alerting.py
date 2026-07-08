"""
脚本 06：告警规则引擎
======================
自动检测实验异常：流量偏移、SRM 告警、归因异常、指标退化、离群值。

告警级别：
- CRITICAL: 阻断 AB 流程（AA 假阳性过高、SRM 不通过）
- WARNING: 需要关注（流量偏移、归因异常）
- INFO: 提示信息（指标退化、离群值）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from typing import List, Dict

from utils.metrics import calculate_all_metrics
from utils.visualize import format_pvalue


def check_aa_failure(retro_aa_status: str, retro_aa_fpr: float) -> List[Dict]:
    """AA 假阳性率过高告警"""
    alerts = []
    if retro_aa_status == 'FAIL':
        alerts.append({
            'rule': 'AA 假阳性率过高',
            'level': 'CRITICAL',
            'detail': f'Retro-AA FPR={retro_aa_fpr:.4f} > 8%，统计系统可能不稳定。'
                      f'建议排查：1)分桶 key 是否正确 2)埋点口径是否一致 3)数据管道过滤规则 4)实验污染',
            'recommendation': '阻断 AB 流程，修复后再实验',
        })
    elif retro_aa_status == 'WARN':
        alerts.append({
            'rule': 'AA 假阳性率偏高',
            'level': 'WARNING',
            'detail': f'Retro-AA FPR={retro_aa_fpr:.4f}，略高于名义 α=0.05',
            'recommendation': '记录风险，关注后续 SRM 检查',
        })
    return alerts


def check_srm(srm_result: dict) -> List[Dict]:
    """SRM 样本比例异常告警"""
    alerts = []
    if not srm_result:
        return alerts

    if not srm_result.get('passed', True):
        alerts.append({
            'rule': 'SRM 样本比例异常',
            'level': 'CRITICAL',
            'detail': f"实际比例={srm_result.get('observed_ratio', '?')}，"
                      f"预期={srm_result.get('expected_ratio', '?')}，"
                      f"p={srm_result.get('p_value', '?')}",
            'recommendation': '检查分桶代码、用户 ID 映射、日志丢失情况',
        })
    return alerts


def check_traffic_skew(df: pd.DataFrame, threshold: float = 0.55) -> List[Dict]:
    """每日流量偏移检查"""
    alerts = []
    for date, day_df in df.groupby('date'):
        counts = day_df['group'].value_counts()
        total = len(day_df)
        if total < 10:  # 样本太小不告警
            continue
        ctrl_pct = counts.get('control', 0) / total
        date_str = str(date)[:10]  # handle both str and datetime
        if ctrl_pct > threshold:
            alerts.append({
                'rule': '流量偏移',
                'level': 'WARNING',
                'detail': f"{date_str}: Control 占比 {ctrl_pct*100:.1f}%（{total} 条），偏离 50%",
                'date': date_str,
                'control_pct': round(ctrl_pct * 100, 1),
            })
        elif ctrl_pct < (1 - threshold):
            alerts.append({
                'rule': '流量偏移',
                'level': 'WARNING',
                'detail': f"{date_str}: Control 占比 {ctrl_pct*100:.1f}%（{total} 条），Treatment 过多",
                'date': date_str,
                'control_pct': round(ctrl_pct * 100, 1),
            })
    return alerts


def check_attribution_anomalies(df: pd.DataFrame) -> List[Dict]:
    """归因异常检查"""
    alerts = []

    # 无曝光有点击
    mask1 = (df['impressions'] == 0) & (df['clicks'] > 0)
    n1 = mask1.sum()
    if n1 > 0:
        alerts.append({
            'rule': '归因异常：无曝光有点击',
            'level': 'WARNING',
            'detail': f'{n1} 行数据曝光=0但点击>0，可能曝光日志丢失',
            'count': int(n1),
        })

    # 无点击有转化
    mask2 = (df['clicks'] == 0) & (df['conversions'] > 0)
    n2 = mask2.sum()
    if n2 > 0:
        alerts.append({
            'rule': '归因异常：无点击有转化',
            'level': 'WARNING',
            'detail': f'{n2} 行数据点击=0但转化>0，归因链路异常',
            'count': int(n2),
        })

    # 无消耗有转化
    mask3 = (df['spend'] == 0) & (df['conversions'] > 0)
    n3 = mask3.sum()
    if n3 > 0:
        alerts.append({
            'rule': '归因异常：无消耗有转化',
            'level': 'WARNING',
            'detail': f'{n3} 行数据消耗=0但转化>0，数据对账问题',
            'count': int(n3),
        })

    # 无消耗有收入
    mask4 = (df['spend'] == 0) & (df['revenue'] > 0)
    n4 = mask4.sum()
    if n4 > 0:
        alerts.append({
            'rule': '归因异常：无消耗有收入',
            'level': 'INFO',
            'detail': f'{n4} 行数据消耗=0但收入>0，自然转化或归因窗口外',
            'count': int(n4),
        })

    return alerts


def check_metric_degradation(
    significance_results: List[Dict],
    threshold: float = 0.10,
) -> List[Dict]:
    """指标退化检查：Treatment 核心指标相对 Control 下降超过阈值"""
    alerts = []
    for r in significance_results:
        if r.get('error'):
            continue
        lift = r.get('lift_pct', 0)
        if lift < -threshold * 100:  # lift_pct 是百分比
            alerts.append({
                'rule': f'{r["metric"]} 显著退化',
                'level': 'WARNING' if r.get('significant') else 'INFO',
                'detail': f"{r['metric']}: {lift:+.1f}% (p={r['p_value']:.4f})",
                'metric': r['metric'],
                'lift_pct': lift,
            })
    return alerts


def check_outliers(df: pd.DataFrame, sigma: float = 3.0) -> List[Dict]:
    """离群值检查：ROAS 超过 Nσ"""
    alerts = []
    if 'ROAS' not in df.columns:
        df = calculate_all_metrics(df)

    roas = df['ROAS'].replace([np.inf, -np.inf], np.nan).dropna()
    if len(roas) == 0:
        return alerts

    mean_roas = roas.mean()
    std_roas = roas.std()
    upper = mean_roas + sigma * std_roas
    lower = mean_roas - sigma * std_roas

    outliers = df[(df['ROAS'] > upper) | (df['ROAS'] < lower)]
    n_outliers = len(outliers)

    if n_outliers > len(df) * 0.02:  # 超过 2% 的记录是离群值
        alerts.append({
            'rule': 'ROAS 离群值过多',
            'level': 'INFO',
            'detail': f'{n_outliers}/{len(df)} 行 ROAS 超过 {sigma}σ '
                      f'(范围: [{lower:.2f}, {upper:.2f}], '
                      f'均值: {mean_roas:.2f}, 标准差: {std_roas:.2f})',
            'n_outliers': int(n_outliers),
            'threshold_lower': round(lower, 2),
            'threshold_upper': round(upper, 2),
        })

    return alerts


def generate_alerts_report(alerts: List[Dict], output_dir: str) -> str:
    """生成告警报告"""
    lines = []
    lines.append("# 实验告警报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    by_level = {'CRITICAL': [], 'WARNING': [], 'INFO': []}
    for a in alerts:
        by_level.get(a.get('level', 'INFO'), []).append(a)

    lines.append(f"## 告警汇总\n")
    lines.append(f"- 🔴 CRITICAL: {len(by_level['CRITICAL'])} 条")
    lines.append(f"- 🟡 WARNING: {len(by_level['WARNING'])} 条")
    lines.append(f"- 🔵 INFO: {len(by_level['INFO'])} 条\n")

    for level in ['CRITICAL', 'WARNING', 'INFO']:
        if by_level[level]:
            icon = {'CRITICAL': '🔴', 'WARNING': '🟡', 'INFO': '🔵'}[level]
            lines.append(f"### {icon} {level}\n")
            for a in by_level[level]:
                lines.append(f"- **{a['rule']}**: {a['detail']}")
                if a.get('recommendation'):
                    lines.append(f"  → {a['recommendation']}")
                lines.append("")
            lines.append("")

    # 保存
    report_path = os.path.join(output_dir, 'alert_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[06_alerting] 告警报告已保存: {report_path}")

    return '\n'.join(lines)


def run_all_checks(
    df: pd.DataFrame,
    config: dict,
    retro_aa_status: str = 'PASS',
    retro_aa_fpr: float = 0.05,
    srm_result: dict = None,
    significance_results: List[Dict] = None,
) -> List[Dict]:
    """运行所有告警规则"""
    alert_config = config.get('alerting', {})

    all_alerts = []

    # AA 检查
    all_alerts.extend(check_aa_failure(retro_aa_status, retro_aa_fpr))

    # SRM 检查
    all_alerts.extend(check_srm(srm_result or {}))

    # 流量偏移
    threshold = alert_config.get('traffic_skew_ratio', 0.55)
    all_alerts.extend(check_traffic_skew(df, threshold))

    # 归因异常
    all_alerts.extend(check_attribution_anomalies(df))

    # 指标退化
    deg_threshold = alert_config.get('metric_degradation_pct', 0.10)
    if significance_results:
        all_alerts.extend(check_metric_degradation(significance_results, deg_threshold))

    # 离群值
    sigma = alert_config.get('outlier_sigma', 3.0)
    all_alerts.extend(check_outliers(df, sigma))

    return all_alerts


def main(
    df: pd.DataFrame,
    config: dict,
    retro_aa_status: str = 'PASS',
    retro_aa_fpr: float = 0.05,
    srm_result: dict = None,
    significance_results: List[Dict] = None,
    output_dir: str = 'data/output',
) -> List[Dict]:
    """主流程"""
    print(f"[06_alerting] 运行告警规则引擎...")

    alerts = run_all_checks(
        df, config, retro_aa_status, retro_aa_fpr,
        srm_result, significance_results,
    )

    # 打印摘要
    by_level = {}
    for a in alerts:
        level = a.get('level', 'INFO')
        by_level[level] = by_level.get(level, 0) + 1

    print(f"[06_alerting] {'='*50}")
    for level in ['CRITICAL', 'WARNING', 'INFO']:
        if by_level.get(level, 0) > 0:
            print(f"  {level}: {by_level[level]} 条")

    if not alerts:
        print(f"  ✓ 无告警，实验运行正常")

    # 生成报告
    generate_alerts_report(alerts, output_dir)

    return alerts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/output/experiment_data.csv')
    parser.add_argument('--output-dir', default='data/output')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    config = {'alerting': {}}
    main(df, config)
