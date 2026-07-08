#!/usr/bin/env python3
"""
============================================================
DSP 广告投放 AB 分桶实验 — 交互式分析工作流
============================================================
逐步执行 → 展示结果 → 手动确认 → 下一步

不是黑盒一键脚本。每一步你都能看到中间结果，
理解数据在哪个环节发生了什么。

用法:
  python run_workflow.py                          # 交互式引导（默认）
  python run_workflow.py --auto                   # 自动模式（调试/CI用）
  python run_workflow.py --step 3                 # 从指定步骤开始
  python run_workflow.py --mode real --group-col experiment_group

面试演示路径：逐步执行，每步讲清楚"这一步在做什么、为什么重要、怎么判读结果"
============================================================
"""

import sys, os, argparse, json, importlib
from datetime import datetime

# 确保 Windows 终端 UTF-8 输出
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 用 importlib 导入数字开头的模块
cleaner = importlib.import_module('scripts.01_data_cleaner')
retro_aa_module = importlib.import_module('scripts.02_retro_aa')
splitter = importlib.import_module('scripts.03_experiment_splitter')
sig_test = importlib.import_module('scripts.04_significance_test')
dash = importlib.import_module('scripts.05_dashboard')
alerts = importlib.import_module('scripts.06_alerting')


# ═══════════════════════════════════════════════════════════
# 交互式 UI 工具
# ═══════════════════════════════════════════════════════════

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def cprint(text, color=''):
    print(f"{color}{text}{Colors.RESET}")

def step_banner(num, total, title, icon=''):
    print(f"\n{'─'*60}")
    print(f"  [{num}/{total}] {icon} {title}")
    print(f"{'─'*60}")

def result_line(label, value, status=''):
    icons = {'ok': '✓', 'warn': '⚠', 'fail': '✗', 'info': '→'}
    icon = icons.get(status, ' ')
    color = {'ok': Colors.GREEN, 'warn': Colors.YELLOW, 'fail': Colors.RED}.get(status, '')
    print(f"  {color}{icon}{Colors.RESET} {label}: {color}{value}{Colors.RESET}")

def ask_continue(default='y'):
    """询问是否继续"""
    options = {
        'y': '继续下一步',
        's': '跳到看板（跳过中间步骤）',
        'q': '退出',
    }
    opts_str = ' / '.join(f"{Colors.BOLD}[{k}]{Colors.RESET}{v}" for k, v in options.items())
    print(f"\n  {opts_str}")
    while True:
        try:
            choice = input(f"  > ").strip().lower()
            if choice in options:
                return choice
            if choice == '':
                return default
        except (EOFError, KeyboardInterrupt):
            return 'q'


# ═══════════════════════════════════════════════════════════
# 各步骤执行函数（返回执行结果供后续步骤使用）
# ═══════════════════════════════════════════════════════════

def run_step1_clean(args):
    """Step 1: 数据清洗"""
    step_banner(1, 6, "数据清洗 & 质量检查", "🧹")
    print()

    df = cleaner.main(args.input, args.output_dir)

    # 结果摘要
    result_line("加载行数", str(len(df)), 'ok')
    result_line("时间跨度", f"{df['date'].min().date()} ~ {df['date'].max().date()}", 'info')
    result_line("媒体平台", f"{df['platform'].nunique()} 个: {', '.join(df['platform'].unique())}", 'info')
    result_line("行业", f"{df['industry'].nunique()} 个", 'info')
    result_line("输出文件", f"{args.output_dir}/cleaned_data.csv", 'ok')

    # 清洗发现
    outlier_cols = [c for c in df.columns if c.endswith('_outlier')]
    n_outliers = sum(df[c].sum() for c in outlier_cols)
    if n_outliers > 0:
        result_line("IQR 异常值", f"{n_outliers} 个数据点", 'warn')

    if 'logic_issue' in df.columns:
        n_logic = df['logic_issue'].notna().sum()
        if n_logic > 0:
            result_line("逻辑异常", f"{n_logic} 行", 'warn')
            for issue_type, count in df['logic_issue'].value_counts().items():
                print(f"    - {issue_type}: {count} 行")

    print(f"\n  📄 清洗详情: {args.output_dir}/data_quality_report.md")
    return df


def run_step2_retro_aa(df, config, args):
    """Step 2: Retro-AA 健康检查"""
    step_banner(2, 6, "Retro-AA 健康检查（AA → AB 方法论核心）", "🏥")
    print()
    print(f"  验证问题: 在没有策略差异时，统计系统会不会报出'显著'？")
    print(f"  方法: 对历史数据做 1000 次随机 50/50 切分 → 每次跑显著性检验")
    print(f"  预期: 只有约 5%（50次）出现显著 → 如果远超 5%，系统有问题")
    print()

    retro_config = config.get('retro_aa', {})
    iterations = retro_config.get('iterations', args.aa_iterations)
    seed = retro_config.get('random_seed', 42)

    status, analysis = retro_aa_module.main(df, config, args.output_dir)

    overall = analysis.get('_overall', {})
    fpr = overall.get('fpr', 0)

    # 结果展示
    icon = {'PASS': 'ok', 'WARN': 'warn', 'FAIL': 'fail'}.get(status, 'info')
    result_line("整体 FPR", f"{fpr:.4f} (名义 α=0.05)", icon)
    result_line("判定", status, icon)

    if status == 'FAIL':
        print(f"\n  {Colors.RED}🔴 AA 未通过！FPR={fpr:.4f} > 8%")
        print(f"  建议阻断 AB 流程，排查:")
        print(f"    1. 分桶 key 是否正确（user ID / device ID / cookie 边界）")
        print(f"    2. 埋点口径是否一致（曝光触发时机、日志漏报）")
        print(f"    3. 数据管道过滤规则（如 bot detection 误杀活跃用户）")
        print(f"    4. 实验污染（跨组用户、多实验互相影响）")
    elif status == 'WARN':
        print(f"\n  {Colors.YELLOW}🟡 AA 略偏高，已记录风险，建议关注后续 SRM 检查。")

    # 分指标详情
    print(f"\n  各指标详情:")
    for metric, info in analysis.items():
        if metric.startswith('_'):
            continue
        s = info.get('status', '')
        print(f"    {metric:8s}  FPR={info['fpr']:.4f}  {s}")

    print(f"\n  📄 AA 报告: {args.output_dir}/retro_aa_report.md")
    return status, analysis


def run_step3_split(df, config, args):
    """Step 3: AB 分桶 + SRM 均衡校验"""
    step_banner(3, 6, "AB 分桶 & SRM 均衡校验", "🎯")
    print()

    df, srm, strata_results, smd_results = splitter.main(df, config, args.output_dir)

    # SRM
    passed = srm.get('passed', False)
    icon = 'ok' if passed else 'fail'
    result_line("SRM 检查", f"p={srm.get('p_value', '?')}  比例={srm.get('observed_ratio', '?')}", icon)

    # 分层均衡
    all_balanced = all(r.get('passed', False) for r in strata_results)
    icon = 'ok' if all_balanced else 'warn'
    result_line("分层均衡", f"{sum(1 for r in strata_results if r.get('passed'))}/{len(strata_results)} 维度通过", icon)

    # SMD
    smd_all_ok = all(abs(r.get('smd', 999)) < 0.1 for r in smd_results)
    icon = 'ok' if smd_all_ok else 'warn'
    result_line("SMD 预实验均衡", f"{sum(1 for r in smd_results if abs(r.get('smd', 999)) < 0.1)}/{len(smd_results)} 指标通过", icon)

    if not all_balanced:
        print(f"\n  {Colors.YELLOW}⚠ 部分维度分布不均衡，可能影响实验结论。")

    print(f"\n  📄 分桶数据: {args.output_dir}/experiment_data.csv")
    print(f"  📄 均衡报告: {args.output_dir}/split_balance_report.md")
    return df, srm, strata_results, smd_results


def run_step4_significance(df, config, args):
    """Step 4: 显著性检验"""
    step_banner(4, 6, "显著性检验", "📊")
    print()

    sig_results, drilldown = sig_test.main(df, config, args.output_dir)

    # 摘要
    n_sig = sum(1 for r in sig_results if r.get('significant'))
    n_fdr_sig = sum(1 for r in sig_results if r.get('fdr_significant'))

    result_line("显著指标", f"{n_sig}/{len(sig_results)} (原始)  {n_fdr_sig}/{len(sig_results)} (FDR校正后)", 'info')

    for r in sig_results:
        if r.get('error'):
            continue
        icon = 'ok' if r.get('significant') else 'info'
        fdr = " ★FDR" if r.get('fdr_significant') else ""
        direction = "↑" if r.get('lift_pct', 0) > 0 else "↓"
        result_line(
            f"{r['metric']}",
            f"{r['control_value']:.4f} → {r['treatment_value']:.4f}  "
            f"({direction}{abs(r['lift_pct']):.1f}%)  "
            f"p={r['p_value']:.4f}{fdr}  "
            f"d={r['effect_size']:.3f}({r.get('effect_magnitude', '')})",
            icon
        )

    if n_sig == 0:
        print(f"\n  ℹ 无显著差异指标。实验组与对照组表现相当。")

    print(f"\n  📄 显著性报告: {args.output_dir}/significance_report.md")
    return sig_results, drilldown


def run_step5_alerts(df, config, retro_aa_status, retro_aa_analysis, srm, sig_results, args):
    """Step 5: 告警扫描"""
    step_banner(5, 6, "告警规则扫描", "🔔")
    print()

    retro_fpr = retro_aa_analysis.get('_overall', {}).get('fpr', 0.05) if retro_aa_analysis else 0.05
    alert_list = alerts.main(
        df, config,
        retro_aa_status=retro_aa_status,
        retro_aa_fpr=retro_fpr,
        srm_result=srm,
        significance_results=sig_results,
        output_dir=args.output_dir,
    )

    by_level = {'CRITICAL': 0, 'WARNING': 0, 'INFO': 0}
    for a in alert_list:
        by_level[a.get('level', 'INFO')] += 1

    if by_level['CRITICAL'] > 0:
        result_line("CRITICAL", f"{by_level['CRITICAL']} 条", 'fail')
    if by_level['WARNING'] > 0:
        result_line("WARNING", f"{by_level['WARNING']} 条", 'warn')
    if by_level['INFO'] > 0:
        result_line("INFO", f"{by_level['INFO']} 条", 'info')
    if sum(by_level.values()) == 0:
        result_line("状态", "无告警，实验运行正常", 'ok')

    for a in alert_list:
        if a.get('level') in ('CRITICAL', 'WARNING'):
            print(f"    [{a['level']}] {a['rule']}: {a['detail']}")

    print(f"\n  📄 告警报告: {args.output_dir}/alert_report.md")
    return alert_list


def run_step6_dashboard(df, config, retro_aa_status, retro_aa_analysis, srm, strata_results,
                         smd_results, sig_results, drilldown, alert_list, args):
    """Step 6: 看板生成"""
    step_banner(6, 6, "实验结论看板生成", "📋")
    print()

    report = dash.main(
        df, config,
        retro_aa_status=retro_aa_status,
        retro_aa_analysis=retro_aa_analysis,
        srm=srm,
        strata_results=strata_results,
        smd_results=smd_results,
        significance_results=sig_results,
        drilldown=drilldown,
        alerts=alert_list,
        output_dir=args.report_dir,
    )

    report_path = os.path.join(args.report_dir, f"experiment_report_{datetime.now().strftime('%Y%m%d')}.md")
    result_line("完整报告", report_path, 'ok')

    print(f"\n  📊 报告包含 6 个章节:")
    print(f"    1. 实验健康检查（AA + SRM）")
    print(f"    2. AB 核心指标对比")
    print(f"    3. 归因分解（根因定位）")
    print(f"    4. 维度下钻分析")
    print(f"    5. 实验结论 & 建议行动")
    print(f"    6. 告警摘要")

    return report


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def print_header():
    print(f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗
║   📊 DSP 广告投放 AB 分桶实验 — 交互式分析工作流           ║
║                                                              ║
║   逐步执行 → 展示结果 → 你决定是否继续                       ║
║   AA→AB 方法论 + 链式归因 + SRM                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
""")


def print_summary(df, retro_aa_status, srm, sig_results, alert_list, report_path):
    print(f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗
║                    🎯 工作流执行完成                          ║
╠══════════════════════════════════════════════════════════════╣
║  数据清洗:     ✓ {len(df):>5d} 行                                        ║
║  Retro-AA:     {retro_aa_status:<10s}                                       ║
║  SRM 校验:     {'✓' if srm and srm.get('passed', False) else '✗':<10s}                                       ║
║  显著指标:     {sum(1 for r in sig_results if r.get('significant')):>3d}/{len(sig_results):<3d}                                      ║
║  告警:         {len(alert_list):>3d} 条                                         ║
╠══════════════════════════════════════════════════════════════╣
║  报告: {report_path}
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
""")


def main():
    parser = argparse.ArgumentParser(description='DSP AB 实验交互式分析工作流')
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--input', default='data/raw/global_ads_performance_dataset.csv')
    parser.add_argument('--output-dir', default='data/output')
    parser.add_argument('--report-dir', default='reports')
    parser.add_argument('--mode', default='simulate', choices=['simulate', 'real'])
    parser.add_argument('--group-col', default='group')
    parser.add_argument('--auto', action='store_true', help='自动模式（非交互）')
    parser.add_argument('--step', type=int, default=1, help='从指定步骤开始（1-6）')
    parser.add_argument('--aa-iterations', type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    print_header()

    config = load_config(args.config)
    config.setdefault('splitting', {})['mode'] = args.mode
    config.setdefault('splitting', {})['group_col'] = args.group_col

    interactive = not args.auto

    # 初始化状态
    df = None
    retro_aa_status = 'NOT_RUN'
    retro_aa_analysis = {}
    srm, strata_results, smd_results = {}, [], []
    sig_results, drilldown = [], {}
    alert_list = []

    # ── Step 1: 数据清洗 ──
    if args.step <= 1:
        df = run_step1_clean(args)
        if interactive:
            c = ask_continue()
            if c == 'q': return
            if c == 's': args.step = max(args.step, 6)  # 跳到看板
    else:
        df = pd.read_csv(os.path.join(args.output_dir, 'cleaned_data.csv'))
        print(f"\n  ⏭ 跳过 Step 1（使用已有数据: {len(df)} 行）")

    # ── Step 2: Retro-AA ──
    if args.step <= 2:
        retro_aa_status, retro_aa_analysis = run_step2_retro_aa(df, config, args)

        if retro_aa_status == 'FAIL':
            print(f"\n  {Colors.RED}建议: 修复分桶/埋点问题后再继续。{Colors.RESET}")
            if interactive:
                c = ask_continue('q')
                if c in ('q', 's'):
                    print("  已退出。请排查问题后重新运行。")
                    return
            else:
                print("  自动模式下继续执行（已标注风险）...")
        elif interactive:
            c = ask_continue()
            if c == 'q': return
            if c == 's': args.step = max(args.step, 6)
    else:
        print(f"\n  ⏭ 跳过 Step 2（Retro-AA）")
        retro_aa_status = 'SKIPPED'

    # ── Step 3: AB 分桶 ──
    if args.step <= 3:
        df, srm, strata_results, smd_results = run_step3_split(df, config, args)
        if interactive:
            c = ask_continue()
            if c == 'q': return
            if c == 's': args.step = max(args.step, 6)
    else:
        df = pd.read_csv(os.path.join(args.output_dir, 'experiment_data.csv'))
        print(f"\n  ⏭ 跳过 Step 3（使用已有分桶数据）")

    # ── Step 4: 显著性检验 ──
    if args.step <= 4:
        sig_results, drilldown = run_step4_significance(df, config, args)
        if interactive:
            c = ask_continue()
            if c == 'q': return
            if c == 's': args.step = max(args.step, 6)
    else:
        print(f"\n  ⏭ 跳过 Step 4")

    # ── Step 5: 告警 ──
    if args.step <= 5:
        alert_list = run_step5_alerts(
            df, config, retro_aa_status, retro_aa_analysis, srm, sig_results, args
        )
        if interactive:
            c = ask_continue()
            if c == 'q': return
    else:
        print(f"\n  ⏭ 跳过 Step 5")

    # ── Step 6: 看板 ──
    report = run_step6_dashboard(
        df, config, retro_aa_status, retro_aa_analysis,
        srm, strata_results, smd_results, sig_results, drilldown, alert_list, args
    )

    report_path = os.path.join(args.report_dir, f"experiment_report_{datetime.now().strftime('%Y%m%d')}.md")
    print_summary(df, retro_aa_status, srm, sig_results, alert_list, report_path)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    main()
