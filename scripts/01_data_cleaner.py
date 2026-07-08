"""
脚本 01：数据清洗 & 质量报告
===========================
读取原始 CSV，统一格式、检测异常、标记问题行，
输出 cleaned_data.csv + 清洗质量报告。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime


def load_data(filepath: str) -> pd.DataFrame:
    """加载原始 CSV 数据"""
    df = pd.read_csv(filepath)
    print(f"[01_data_cleaner] 加载数据: {len(df)} 行, {len(df.columns)} 列")
    return df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """统一日期格式"""
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    bad_dates = df['date'].isna().sum()
    if bad_dates > 0:
        print(f"  ⚠ 发现 {bad_dates} 行日期格式异常，已标记")
    return df


def check_missing(df: pd.DataFrame) -> dict:
    """缺失值检查"""
    missing = {}
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            missing[col] = n_missing
            print(f"  ⚠ 列 '{col}' 缺失 {n_missing} 行 ({n_missing/len(df)*100:.1f}%)")
    if not missing:
        print("  ✓ 所有列无缺失值")
    return missing


def detect_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """IQR 方法检测异常值，添加标记列"""
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        flag_col = f'{col}_outlier'
        df[flag_col] = ~df[col].between(lower, upper)
        n_outliers = df[flag_col].sum()
        if n_outliers > 0:
            print(f"  ⚠ '{col}' 发现 {n_outliers} 个 IQR 异常值 (范围: [{lower:.2f}, {upper:.2f}])")

    return df


def logic_checks(df: pd.DataFrame) -> pd.DataFrame:
    """逻辑一致性校验"""
    issues = []

    # 有点击但无曝光
    mask1 = (df['clicks'] > 0) & (df['impressions'] == 0)
    if mask1.sum() > 0:
        issues.append(f"{mask1.sum()} 行: 有点击但无曝光（可能日志丢失）")
        df.loc[mask1, 'logic_issue'] = 'clicks_without_impressions'

    # 有转化但无点击
    mask2 = (df['conversions'] > 0) & (df['clicks'] == 0)
    if mask2.sum() > 0:
        issues.append(f"{mask2.sum()} 行: 有转化但无点击（归因异常）")
        df.loc[mask2, 'logic_issue'] = 'conversions_without_clicks'

    # 有消耗但无点击
    mask3 = (df['spend'] if 'spend' in df.columns else df['ad_spend']) > 0
    mask3 = mask3 & (df['clicks'] == 0)
    if mask3.sum() > 0:
        issues.append(f"{mask3.sum()} 行: 有消耗但无点击（低效投放）")
        df.loc[mask3, 'logic_issue'] = 'spend_without_clicks'

    # 无消耗但有转化
    spend_col = 'spend' if 'spend' in df.columns else 'ad_spend'
    mask4 = (df[spend_col] == 0) & (df['conversions'] > 0)
    if mask4.sum() > 0:
        issues.append(f"{mask4.sum()} 行: 无消耗但有转化（数据异常）")
        df.loc[mask4, 'logic_issue'] = 'conversions_without_spend'

    if not issues:
        print("  ✓ 逻辑校验全部通过")
    else:
        for issue in issues:
            print(f"  ⚠ {issue}")

    return df


def generate_quality_report(df: pd.DataFrame, missing: dict, output_dir: str) -> str:
    """生成数据质量报告"""
    lines = []
    lines.append("# 数据清洗质量报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 数据概况")
    lines.append(f"- 总行数: {len(df)}")
    lines.append(f"- 总列数: {len(df.columns)}")
    lines.append(f"- 时间跨度: {df['date'].min().date()} ~ {df['date'].max().date()}")
    lines.append(f"- 媒体平台: {df['platform'].nunique()} 个 ({', '.join(df['platform'].unique())})")
    lines.append(f"- 行业: {df['industry'].nunique()} 个 ({', '.join(df['industry'].unique())})")
    lines.append(f"- 国家: {df['country'].nunique()} 个 ({', '.join(df['country'].unique())})\n")

    lines.append("## 缺失值检查")
    if missing:
        for col, n in missing.items():
            lines.append(f"- `{col}`: {n} 行缺失 ({n/len(df)*100:.1f}%)")
    else:
        lines.append("- ✓ 所有列无缺失值")

    lines.append("\n## 异常值检查")
    outlier_cols = [c for c in df.columns if c.endswith('_outlier')]
    for col in outlier_cols:
        n = df[col].sum()
        if n > 0:
            base = col.replace('_outlier', '')
            lines.append(f"- `{base}`: {n} 个 IQR 异常值")

    lines.append("\n## 逻辑校验")
    if 'logic_issue' in df.columns:
        issues = df['logic_issue'].dropna()
        if len(issues) > 0:
            for issue_type, count in issues.value_counts().items():
                lines.append(f"- `{issue_type}`: {count} 行")
        else:
            lines.append("- ✓ 无逻辑异常")
    else:
        lines.append("- ✓ 无逻辑异常")

    quality_path = os.path.join(output_dir, 'data_quality_report.md')
    with open(quality_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n[01_data_cleaner] 质量报告已保存: {quality_path}")

    return '\n'.join(lines)


def main(raw_path: str, output_dir: str):
    """主流程"""
    # 1. 加载
    df = load_data(raw_path)

    # 2. 日期清洗
    df = clean_dates(df)

    # 3. 缺失值检查
    missing = check_missing(df)

    # 4. 异常值检测
    numeric_cols = ['impressions', 'clicks', 'ad_spend', 'spend', 'conversions', 'revenue']
    df = detect_outliers_iqr(df, [c for c in numeric_cols if c in df.columns])

    # 5. 逻辑校验
    df = logic_checks(df)

    # 6. 统一列名
    if 'ad_spend' in df.columns:
        df['spend'] = df['ad_spend']

    # 7. 保存清洗后数据
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cleaned_data.csv')
    df.to_csv(output_path, index=False)
    print(f"[01_data_cleaner] 清洗后数据已保存: {output_path}")

    # 8. 生成质量报告
    report = generate_quality_report(df, missing, output_dir)

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DSP数据清洗')
    parser.add_argument('--input', default='data/raw/global_ads_performance_dataset.csv')
    parser.add_argument('--output-dir', default='data/output')
    args = parser.parse_args()

    df = main(args.input, args.output_dir)
    print(f"\n[01_data_cleaner] ✓ 完成，清洗后 {len(df)} 行")
