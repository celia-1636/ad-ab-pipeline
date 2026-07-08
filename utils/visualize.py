"""
可视化工具模块
===============
ASCII 图表、文本可视化、Mermaid 图表生成。
用于 Markdown 报告中的可视化展示。
"""

import numpy as np
from typing import List, Tuple, Optional


def ascii_bar_chart(
    items: List[Tuple[str, float, str]],
    max_width: int = 40,
    title: str = "",
    sort: bool = True,
) -> str:
    """
    生成 ASCII 条形图。

    Parameters:
    -----------
    items : List[Tuple[str, float, str]]
        [(标签, 数值, 颜色/注释), ...]，数值的符号决定方向
    max_width : int
        最大条形宽度（字符数）
    title : str
        图表标题
    sort : bool
        是否按绝对值降序排列

    Returns:
    --------
    str: ASCII 条形图字符串
    """
    if not items:
        return "(无数据)"

    if sort:
        items = sorted(items, key=lambda x: abs(x[1]), reverse=True)

    # 找到最大绝对值用于缩放
    max_abs = max(abs(v) for _, v, _ in items)
    if max_abs == 0:
        max_abs = 1

    lines = []
    if title:
        lines.append(title)
        lines.append("─" * (max_width + 30))

    # 找到最长标签名用于对齐
    max_label_len = max(len(label) for label, _, _ in items)

    for label, value, note in items:
        bar_len = int(abs(value) / max_abs * max_width)
        bar = "█" * bar_len
        sign = "+" if value > 0 else "-" if value < 0 else " "
        lines.append(
            f"  {label:<{max_label_len}} {sign}{bar:<{max_width}} {note}"
        )

    return "\n".join(lines)


def ascii_waterfall(
    items: List[Tuple[str, float, float, float]],
    total_start: float,
    total_end: float,
    title: str = "",
) -> str:
    """
    生成贡献瀑布图（ASCII 格式）。

    Parameters:
    -----------
    items : List[Tuple[str, float, float, float]]
        [(因子名, 基准值, 实验值, 贡献值), ...]
    total_start : float
        起点总值
    total_end : float
        终点总值
    title : str
        图表标题

    Returns:
    --------
    str: ASCII 瀑布图字符串
    """
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * 70)

    lines.append(f"{'起点':20s} {total_start:>15,.2f}")
    lines.append("-" * 40)

    running = total_start
    for name, v0, v1, contrib in items:
        direction = "↑" if contrib > 0 else "↓" if contrib < 0 else "→"
        lines.append(
            f"{name:20s} {direction} {contrib:>+13,.2f}  "
            f"({v0:,.4f} → {v1:,.4f})"
        )
        running += contrib

    lines.append("-" * 40)
    lines.append(f"{'终点':20s} {total_end:>15,.2f}")
    lines.append(f"{'总变化':20s} {total_end - total_start:>+15,.2f}")

    return "\n".join(lines)


def format_pvalue(p: float, alpha: float = 0.05) -> str:
    """
    格式化 p-value，添加显著性标记。

    Returns:
    --------
    str: 如 "0.0032 ***" 或 "0.14 ns"
    """
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    elif p < 0.1:
        stars = "."
    else:
        stars = "ns"

    # 格式化 p 值
    if p < 0.0001:
        p_str = f"{p:.2e}"
    elif p < 0.001:
        p_str = f"{p:.4f}"
    elif p < 0.01:
        p_str = f"{p:.4f}"
    elif p < 1:
        p_str = f"{p:.4f}"
    else:
        p_str = f"{p:.4f}"

    return f"{p_str} {stars}"


def format_pct_change(v0: float, v1: float) -> str:
    """格式化百分比变化"""
    if v0 == 0:
        return "N/A"
    pct = (v1 - v0) / abs(v0) * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.2f}%"


def format_smd(smd: float) -> str:
    """格式化 SMD 值并判断是否均衡"""
    if abs(smd) < 0.1:
        return f"{smd:.4f} ✓ (均衡)"
    elif abs(smd) < 0.2:
        return f"{smd:.4f} ⚠ (轻微偏差)"
    else:
        return f"{smd:.4f} ✗ (明显偏差)"


def fpr_status(fpr: float, pass_threshold: float = 0.06, warn_threshold: float = 0.08) -> str:
    """判断 Retro-AA 假阳性率状态"""
    if fpr <= pass_threshold:
        return "🟢 PASS"
    elif fpr <= warn_threshold:
        return "🟡 WARN"
    else:
        return "🔴 FAIL"


def pvalue_distribution_histogram(pvalues: List[float], bins: int = 10, title: str = "p-value 分布") -> str:
    """
    生成 p-value 分布的 ASCII 直方图。
    用于 Retro-AA 检查 p-value 是否均匀分布。
    """
    if not pvalues:
        return "(无数据)"

    hist, edges = np.histogram(pvalues, bins=bins, range=(0, 1))
    max_count = max(hist) if max(hist) > 0 else 1
    max_width = 30

    lines = [title, "─" * 50]
    lines.append(f"{'区间':<12s} {'频数':>6s} {'分布':>}")
    lines.append("-" * 50)

    for i in range(bins):
        left = edges[i]
        right = edges[i + 1]
        count = hist[i]
        bar_len = int(count / max_count * max_width)
        bar = "█" * bar_len
        lines.append(f"[{left:.1f}-{right:.1f}]  {count:>6d}  {bar}")

    # 期望线
    expected = len(pvalues) / bins
    lines.append("-" * 50)
    lines.append(f"期望每区间: {expected:.1f}")
    lines.append("理想分布: 均匀分布（各区间的柱子高度应大致相等）")
    lines.append("警告信号: p-value 集中在 [0.0-0.1] → 统计系统可能在无差异时制造差异")

    return "\n".join(lines)
