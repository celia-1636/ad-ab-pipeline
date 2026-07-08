"""
DSP 广告主侧归因分解模块
=========================
两类指标的根因定位：

1. 乘法型指标 → 链式分解（Chain Rule）
   Revenue = Impressions × CTR × CVR × ARPConv
   Spend = Impressions × CPM / 1000

2. 除法型指标 → 差分分解（Difference Decomposition）
   CTR = Clicks / Impressions
   ROAS = Revenue / Spend
   等等

视角：DSP 广告主侧，关注花了多少钱、拿到多少效果。
"""

import numpy as np
from typing import Dict, List, Tuple


def chain_decompose(
    factors: List[Tuple[str, float, float]],
    label: str = "G"
) -> Dict:
    """
    链式分解（Chain Rule）：将乘法指标 G = A × B × C × D 的变化
    归因到各因子。

    Parameters:
    -----------
    factors : List[Tuple[str, float, float]]
        因子列表，每个元素为 (因子名, 基准值, 实验值)
        顺序决定归因结果（应按业务漏斗顺序排列）
    label : str
        指标名称

    Returns:
    --------
    dict: {
        'label': 指标名,
        'G0': 基准总值,
        'G1': 实验总值,
        'delta': 总变化,
        'contributions': [(因子名, 贡献值, 贡献占比), ...],
        'residual': 未解释残差
    }
    """
    if len(factors) < 2:
        raise ValueError("至少需要 2 个因子")

    # 计算基准总值和实验总值
    G0 = 1.0
    G1 = 1.0
    for name, v0, v1 in factors:
        G0 *= v0
        G1 *= v1

    delta = G1 - G0

    contributions = []
    explained = 0.0

    for i, (name, v0, v1) in enumerate(factors):
        # 链式分解公式：
        # 因子 i 的贡献 = (前面因子用实验值) × (当前因子差值) × (后面因子用基准值)
        contrib = 1.0
        for j in range(i):
            contrib *= factors[j][2]  # 前面的因子用实验值
        contrib *= (v1 - v0)          # 当前因子的变化
        for j in range(i + 1, len(factors)):
            contrib *= factors[j][1]  # 后面的因子用基准值

        explained += contrib
        pct = contrib / delta * 100 if delta != 0 else 0
        contributions.append((name, v0, v1, contrib, pct))

    residual = delta - explained

    return {
        'label': label,
        'G0': G0,
        'G1': G1,
        'delta': delta,
        'delta_pct': delta / G0 * 100 if G0 != 0 else 0,
        'contributions': contributions,
        'residual': residual,
    }


def diff_decompose(
    numerator_name: str,
    N0: float, N1: float,
    D0: float, D1: float,
    label: str = "R"
) -> Dict:
    """
    差分分解（Difference Decomposition）：将除法指标 R = N/D 的变化
    拆解为分子贡献和分母贡献。

    公式：
    ΔR ≈ (ΔN / D₀) - (N₀ × ΔD / D₀²)

    Parameters:
    -----------
    numerator_name : str
        分子名称
    N0, N1 : float
        基准期和实验期的分子值
    D0, D1 : float
        基准期和实验期的分母值
    label : str
        指标名称

    Returns:
    --------
    dict: {
        'label': 指标名,
        'N0', 'N1': 分子值,
        'D0', 'D1': 分母值,
        'R0': 基准比率,
        'R1': 实验比率,
        'delta': 比率变化,
        'numerator_contrib': 分子贡献,
        'denominator_contrib': 分母贡献,
        'residual': 未解释残差
    }
    """
    R0 = N0 / D0 if D0 != 0 else 0
    R1 = N1 / D1 if D1 != 0 else 0
    delta = R1 - R0

    # 差分分解
    dN = N1 - N0
    dD = D1 - D0

    # 分子贡献 = ΔN / D₀
    num_contrib = dN / D0 if D0 != 0 else 0

    # 分母贡献 = -N₀ × ΔD / D₀²
    denom_contrib = -(N0 * dD) / (D0 * D0) if D0 != 0 else 0

    explained = num_contrib + denom_contrib
    residual = delta - explained

    # 贡献占比
    total_abs = abs(num_contrib) + abs(denom_contrib)
    num_pct = abs(num_contrib) / total_abs * 100 if total_abs != 0 else 0
    denom_pct = abs(denom_contrib) / total_abs * 100 if total_abs != 0 else 0

    return {
        'label': label,
        'N0': N0, 'N1': N1,
        'D0': D0, 'D1': D1,
        'R0': R0, 'R1': R1,
        'delta': delta,
        'delta_pp': delta * 100 if R0 < 1 else delta,  # 比率用百分点
        'numerator_contrib': num_contrib,
        'denominator_contrib': denom_contrib,
        'numerator_pct': num_pct,
        'denominator_pct': denom_pct,
        'residual': residual,
        'numerator_name': numerator_name,
        'denominator_name': 'denominator',
    }


def decompose_revenue(
    ctrl_impressions: float, treat_impressions: float,
    ctrl_clicks: float, treat_clicks: float,
    ctrl_conversions: float, treat_conversions: float,
    ctrl_revenue: float, treat_revenue: float,
) -> Dict:
    """
    Revenue 归因分解：Revenue = Impressions × CTR × CVR × ARPConv

    从聚合级别的总数进行链式分解。
    """
    # 计算各组比率
    ctrl_ctr = ctrl_clicks / ctrl_impressions if ctrl_impressions > 0 else 0
    treat_ctr = treat_clicks / treat_impressions if treat_impressions > 0 else 0

    ctrl_cvr = ctrl_conversions / ctrl_clicks if ctrl_clicks > 0 else 0
    treat_cvr = treat_conversions / treat_clicks if treat_clicks > 0 else 0

    ctrl_arpconv = ctrl_revenue / ctrl_conversions if ctrl_conversions > 0 else 0
    treat_arpconv = treat_revenue / treat_conversions if treat_conversions > 0 else 0

    factors = [
        ('曝光量', ctrl_impressions, treat_impressions),
        ('CTR', ctrl_ctr, treat_ctr),
        ('CVR', ctrl_cvr, treat_cvr),
        ('ARPConv', ctrl_arpconv, treat_arpconv),
    ]

    return chain_decompose(factors, label='Revenue')


def decompose_roas(
    ctrl_revenue: float, treat_revenue: float,
    ctrl_spend: float, treat_spend: float,
) -> Dict:
    """
    ROAS 归因分解：ROAS = Revenue / Spend
    """
    return diff_decompose(
        'Revenue', ctrl_revenue, treat_revenue,
        ctrl_spend, treat_spend,
        label='ROAS'
    )


def decompose_ratio_metric(
    metric_name: str,
    ctrl_num: float, treat_num: float,
    ctrl_den: float, treat_den: float,
) -> Dict:
    """
    通用比率指标分解（CTR/CVR/CPC/CPM/CPA/ROAS）
    """
    return diff_decompose(
        metric_name.split('/')[0] if '/' not in metric_name else metric_name,
        ctrl_num, treat_num,
        ctrl_den, treat_den,
        label=metric_name
    )
