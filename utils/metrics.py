"""
DSP 广告主侧核心指标计算模块
==============================
计算 CTR、CVR、CPC、CPM、CPA、ROAS 等广告主侧核心指标。

视角说明：这是 DSP 广告主侧分析，关注广告主花了多少钱、拿到多少效果。
不是媒体变现侧（eCPM、填充率是媒体平台看的）。
"""

import numpy as np
import pandas as pd


def safe_divide(numerator, denominator, default=0.0):
    """安全除法，避免除零"""
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    为 DataFrame 补充计算所有广告主侧核心指标。

    输入 DataFrame 需包含以下列：
    - impressions: 曝光量
    - clicks: 点击量
    - conversions: 转化量
    - ad_spend / spend: 消耗
    - revenue: 转化营收

    返回添加了以下列的 DataFrame：
    - CTR: 点击率 = clicks / impressions
    - CVR: 转化率 = conversions / clicks
    - CPC: 单次点击成本 = spend / clicks
    - CPM: 千次曝光成本 = spend / impressions * 1000
    - CPA: 单次转化成本 = spend / conversions
    - ROAS: 广告支出回报率 = revenue / spend
    - ARPConv: 单次转化收入 = revenue / conversions
    """
    df = df.copy()

    # 统一 spend 列名
    if 'ad_spend' in df.columns and 'spend' not in df.columns:
        df['spend'] = df['ad_spend']

    # --- 比率指标 ---
    # CTR: 点击率
    df['CTR'] = df.apply(
        lambda r: safe_divide(r['clicks'], r['impressions']), axis=1
    )

    # CVR: 转化率
    df['CVR'] = df.apply(
        lambda r: safe_divide(r['conversions'], r['clicks']), axis=1
    )

    # CPC: 单次点击成本
    df['CPC'] = df.apply(
        lambda r: safe_divide(r['spend'], r['clicks']), axis=1
    )

    # CPM: 千次曝光成本
    df['CPM'] = df.apply(
        lambda r: safe_divide(r['spend'] * 1000, r['impressions']), axis=1
    )

    # CPA: 单次转化成本
    df['CPA'] = df.apply(
        lambda r: safe_divide(r['spend'], r['conversions']), axis=1
    )

    # ROAS: 广告支出回报率
    df['ROAS'] = df.apply(
        lambda r: safe_divide(r['revenue'], r['spend']), axis=1
    )

    # ARPConv: 单次转化收入（Average Revenue Per Conversion）
    df['ARPConv'] = df.apply(
        lambda r: safe_divide(r['revenue'], r['conversions']), axis=1
    )

    return df


def aggregate_metrics(df: pd.DataFrame, group_col: str = 'group') -> pd.DataFrame:
    """
    按分组聚合计算汇总指标。

    返回包含各组汇总值的 DataFrame：
    - total_impressions, total_clicks, total_conversions
    - total_spend, total_revenue
    - 以及衍生的比率指标
    """
    agg = df.groupby(group_col).agg(
        total_impressions=('impressions', 'sum'),
        total_clicks=('clicks', 'sum'),
        total_conversions=('conversions', 'sum'),
        total_spend=('spend', 'sum'),
        total_revenue=('revenue', 'sum'),
        n_rows=('impressions', 'count'),
    ).reset_index()

    # 从汇总值重新计算比率（更准确，避免 Simpson 悖论）
    agg['CTR'] = agg.apply(
        lambda r: safe_divide(r['total_clicks'], r['total_impressions']), axis=1
    )
    agg['CVR'] = agg.apply(
        lambda r: safe_divide(r['total_conversions'], r['total_clicks']), axis=1
    )
    agg['CPC'] = agg.apply(
        lambda r: safe_divide(r['total_spend'], r['total_clicks']), axis=1
    )
    agg['CPM'] = agg.apply(
        lambda r: safe_divide(r['total_spend'] * 1000, r['total_impressions']), axis=1
    )
    agg['CPA'] = agg.apply(
        lambda r: safe_divide(r['total_spend'], r['total_conversions']), axis=1
    )
    agg['ROAS'] = agg.apply(
        lambda r: safe_divide(r['total_revenue'], r['total_spend']), axis=1
    )
    agg['ARPConv'] = agg.apply(
        lambda r: safe_divide(r['total_revenue'], r['total_conversions']), axis=1
    )

    return agg
