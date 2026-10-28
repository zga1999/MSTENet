"""绘制「不同 IoU 阈值下的定位性能」折线图（Baseline vs +HPD）。

只依赖 numpy + matplotlib，AP 数据直接写在下面的 BASELINE / METHOD 里，
换数据时改常量或用命令行传入即可。

用法一：用文件里的默认数据出图::

    python tools/analysis_tools/plot_ap_vs_iou.py --out-dir work_vis/ap_vs_iou

用法二：命令行传入自己的 AP 序列（长度需与 --thresholds 一致）::

    python tools/analysis_tools/plot_ap_vs_iou.py \
        --baseline 85.0 81.7 76.8 71.2 64.6 56.8 47.4 36.1 20.9 6.0 \
        --method 85.4 82.2 77.9 72.6 66.4 59.3 50.8 40.0 25.2 8.1 \
        --method-label '+ HPD' \
        --annotate-at 0.85
"""

import argparse
import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
BASELINE = [85.0, 81.7, 76.8, 71.2, 64.6, 56.8, 47.4, 36.1, 20.9, 6.0]
METHOD = [85.4, 82.2, 77.9, 72.6, 66.4, 59.3, 50.8, 40.0, 25.2, 8.1]

TITLE = 'Localization Performance under Different IoU Thresholds'
XLABEL = 'IoU threshold'
YLABEL = 'AP (%)'
ANNOTATION = 'Larger gains at higher IoU thresholds'

COLOR_BASELINE = '#1f77b4'
COLOR_METHOD = '#ff7f0e'

# 优先级从高到低的 Times 系字体；容器里没装 Times New Roman 时自动降级
_SERIF_CANDIDATES = [
    'Times New Roman', 'Nimbus Roman', 'Nimbus Roman No9 L',
    'Liberation Serif', 'Tinos', 'FreeSerif', 'DejaVu Serif',
]


def resolve_serif_font(font_file=None):
    """挑一个可用的 Times 系字体，font_file 给了就直接注册使用。"""
    if font_file:
        fm.fontManager.addfont(font_file)
        name = fm.FontProperties(fname=font_file).get_name()
        print(f'[✓] 已加载字体文件 {font_file} -> "{name}"')
        return name

    available = {f.name for f in fm.fontManager.ttflist}
    for name in _SERIF_CANDIDATES:
        if name in available:
            if name != 'Times New Roman':
                print(f'[!] 未找到 Times New Roman，改用 "{name}"。'
                      f'如需严格 Times：加 --font-file /path/to/times.ttf')
            return name
    return 'DejaVu Serif'


def apply_paper_style(font_size=13.0, font_file=None):
    font = resolve_serif_font(font_file)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [font] + _SERIF_CANDIDATES,
        'mathtext.fontset': 'stix',
        'font.size': font_size,
        'axes.titlesize': font_size + 2,
        'axes.labelsize': font_size + 1,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size - 1,
        'axes.unicode_minus': False,
        'axes.linewidth': 1.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })


def plot_ap_vs_iou(thresholds,
                   baseline,
                   method,
                   baseline_label='Baseline',
                   method_label='+ HPD',
                   annotate_at=0.85,
                   figsize=(7.2, 5.4)):
    thresholds = np.asarray(thresholds, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    method = np.asarray(method, dtype=np.float64)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(thresholds, baseline, '-o', color=COLOR_BASELINE, lw=2.0,
            ms=7, mec='none', label=baseline_label, zorder=3)
    ax.plot(thresholds, method, '-s', color=COLOR_METHOD, lw=2.0,
            ms=7, mec='none', label=method_label, zorder=3)

    ax.set_title(TITLE, pad=12)
    ax.set_xlabel(XLABEL, labelpad=8)
    ax.set_ylabel(YLABEL, labelpad=8)

    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax.set_xlim(thresholds[0] - 0.02, thresholds[-1] + 0.02)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(True, ls='--', lw=0.6, color='0.85', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=True, framealpha=1.0,
              edgecolor='0.7', borderpad=0.5)

    # 箭头标注：指向指定 IoU 阈值上本方法的数据点
    if annotate_at is not None:
        idx = int(np.argmin(np.abs(thresholds - annotate_at)))
        x, y = thresholds[idx], method[idx]
        # 文字贴右边框，箭头从文字框左下角斜指向数据点（避免压住曲线）
        ax.annotate(ANNOTATION,
                    xy=(x, y + 1.5),
                    xytext=(thresholds[-1] + 0.012, y + 12),
                    ha='right', va='center',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2,
                                    shrinkA=4, shrinkB=2))

    fig.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description='绘制不同 IoU 阈值下的 AP 折线图')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=THRESHOLDS, help='IoU 阈值序列')
    parser.add_argument('--baseline', type=float, nargs='+',
                        default=BASELINE, help='baseline 各阈值下的 AP (%%)')
    parser.add_argument('--method', type=float, nargs='+',
                        default=METHOD, help='本方法各阈值下的 AP (%%)')
    parser.add_argument('--baseline-label', default='Baseline')
    parser.add_argument('--method-label', default='+ HPD')
    parser.add_argument('--annotate-at', type=float, default=-1,
                        help='箭头标注指向的 IoU 阈值，默认不画；传 0.85 等则画标注')
    parser.add_argument('--out-dir', default='work_vis/ap_vs_iou')
    parser.add_argument('--tag', default='ap_vs_iou', help='输出文件名前缀')
    parser.add_argument('--font-size', type=float, default=13.0)
    parser.add_argument('--font-file', default=None,
                        help='times.ttf 等字体文件路径（可选）')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    n = len(args.thresholds)
    if len(args.baseline) != n or len(args.method) != n:
        parser.error(f'--baseline/--method 长度需与 --thresholds ({n}) 一致')
    return args


def main():
    args = parse_args()
    apply_paper_style(args.font_size, args.font_file)

    fig = plot_ap_vs_iou(
        args.thresholds, args.baseline, args.method,
        baseline_label=args.baseline_label,
        method_label=args.method_label,
        annotate_at=None if args.annotate_at < 0 else args.annotate_at)

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        out = osp.join(args.out_dir, f'{args.tag}.{ext}')
        fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
        print(f'[✓] 已保存 {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
