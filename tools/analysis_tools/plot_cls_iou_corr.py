"""按论文风格绘制「分类分数 vs IoU」密度散点图（参考图第 2 行样式）。

本文件只依赖 numpy + matplotlib（scipy 可选，用于 KDE），
不依赖 mmcv / mmdet，因此可以在任意环境里直接对 .npz 重新出图，
方便反复调样式而不用重跑模型推理。

用法一：由 vis_cls_reg_correlation.py 收集完数据后自动调用。

用法二：拿已缓存的 .npz 单独重画（推荐调样式时用）::

    python tools/analysis_tools/plot_cls_iou_corr.py \
        work_vis/cls_reg_correlation/raw_dota_val.npz \
        --out-dir work_vis/cls_reg_correlation \
        --tag dota_val \
        --drop-degenerate \
        --max-dots 20000
"""

import argparse
import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.transforms import Bbox

# density 样式的横轴文字（含子图编号，方便直接放进论文）
XLABEL_IN = '(a) IoU between anchors and GT boxes'
XLABEL_OUT = '(b) IoU between predicted and GT boxes'
YLABEL = 'Classification score $s$'

# dots 样式（CFC-Net 风格）：子图编号单独放在最下面，横轴只写说明文字
XLABEL_IN_PLAIN = 'IoU between anchors and GT boxes'
XLABEL_OUT_PLAIN = 'IoU between predicted boxes and GT boxes'
YLABEL_CONF = 'Classification Confidence'

# 优先级从高到低的 Times 系字体；容器里没装 Times New Roman 时自动降级
_SERIF_CANDIDATES = [
    'Times New Roman', 'Nimbus Roman', 'Nimbus Roman No9 L',
    'Liberation Serif', 'Tinos', 'FreeSerif', 'DejaVu Serif',
]


# ════════════════════════════════════════════════════════════════════════
#  统计量
# ════════════════════════════════════════════════════════════════════════
def rankdata(a):
    """平均秩（与 scipy.stats.rankdata 的 'average' 一致），全向量化。"""
    a = np.asarray(a, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank = cum - (counts - 1) / 2.0
    return avg_rank[inv]


def spearman(a, b):
    """Spearman 秩相关系数 = 秩上的 Pearson 相关。"""
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a, np.float64),
                             np.asarray(b, np.float64))[0, 1])


# ════════════════════════════════════════════════════════════════════════
#  点密度估计（决定每个散点的颜色深浅）
# ════════════════════════════════════════════════════════════════════════
def point_density(x, y, method='kde', kde_fit_max=6000, bins=90, seed=0):
    """返回每个样本点处的密度值，长度与 x 相同。

    method='kde'  : 高斯核密度，颜色过渡最平滑（参考图的效果）。
                    为控制耗时，只用随机 kde_fit_max 个点拟合核，
                    再在全部点上求值。
    method='hist' : 二维直方图计数，不需要 scipy，速度快。
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)

    if method == 'kde':
        try:
            from scipy.stats import gaussian_kde
        except ImportError:
            print('[!] 未安装 scipy，密度估计自动降级为 hist')
            method = 'hist'

    if method == 'kde':
        rng = np.random.default_rng(seed)
        fit_idx = rng.choice(len(x), size=min(len(x), kde_fit_max),
                             replace=False)
        kde = gaussian_kde(np.vstack([x[fit_idx], y[fit_idx]]))
        return kde(np.vstack([x, y]))

    hist, xe, ye = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])
    # 3x3 均值平滑，避免格子边界出现色块断层
    pad = np.pad(hist, 1, mode='edge')
    hist = sum(pad[i:i + bins, j:j + bins]
               for i in range(3) for j in range(3)) / 9.0
    ix = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
    return hist[ix, iy]


# ════════════════════════════════════════════════════════════════════════
#  样式
# ════════════════════════════════════════════════════════════════════════
def resolve_serif_font(font_file=None):
    """挑一个可用的 Times 系字体。

    font_file: 指向 .ttf/.otf 的路径（例如从 Windows 拷来的 times.ttf），
    给了就直接注册使用，这是拿到严格 Times New Roman 最省事的办法。
    """
    if font_file:
        fm.fontManager.addfont(font_file)
        name = fm.FontProperties(fname=font_file).get_name()
        print(f'[✓] 已加载字体文件 {font_file} -> "{name}"')
        return name

    available = {f.name for f in fm.fontManager.ttflist}
    for name in _SERIF_CANDIDATES:
        if name in available:
            if name == 'Times New Roman':
                return name
            metric_compatible = name in ('Nimbus Roman', 'Nimbus Roman No9 L',
                                         'Liberation Serif', 'Tinos')
            note = '与 Times 度量兼容' if metric_compatible else '字形与 Times 有差异'
            print(f'[!] 未找到 Times New Roman，改用 "{name}"（{note}）。\n'
                  f'    如需严格 Times：把 times.ttf 拷到本机后加 '
                  f'--font-file /path/to/times.ttf，或 '
                  f'cp times.ttf ~/.fonts/ && rm -rf ~/.cache/matplotlib')
            return name
    return 'DejaVu Serif'


def apply_paper_style(font_size=10.5, font_file=None):
    """论文排版：Times 系字体 + 5 号字（10.5 pt）+ 细边框。"""
    font = resolve_serif_font(font_file)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [font] + _SERIF_CANDIDATES,
        'mathtext.fontset': 'stix',          # 数学符号（ρ、s）与 Times 最接近
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'axes.unicode_minus': False,
        'axes.linewidth': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,                  # 投稿要求：字体可编辑而非曲线化
        'ps.fonttype': 42,
    })
    return font


def save_figure(fig, png_path, dpi=600, pad=0.06, save_pdf=True):
    """保存图像，并保证长轴标签不被裁掉。

    matplotlib 的 bbox_inches='tight' 会把 x 轴标签的宽度截到坐标系宽度
    （防止 constrained_layout 抖动），标签比坐标系宽时右侧文字就会丢。
    这里手动把所有文字对象的包围盒并进去再裁。
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
    boxes += [t.get_window_extent(renderer) for t in fig.findobj(mtext.Text)
              if t.get_visible() and t.get_text().strip()]
    boxes = [b for b in boxes if b is not None and b.width > 0 and b.height > 0]
    bb = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)

    fig.savefig(png_path, dpi=dpi, bbox_inches=bb)
    if save_pdf:
        fig.savefig(osp.splitext(png_path)[0] + '.pdf', bbox_inches=bb)


def blue_cmap():
    """参考图的浅蓝→深蓝渐变。"""
    return LinearSegmentedColormap.from_list(
        'paper_blues',
        ['#dbeaf7', '#9ecae1', '#4a97c9', '#1f6cb0', '#08306b'])


# ════════════════════════════════════════════════════════════════════════
#  单个面板
# ════════════════════════════════════════════════════════════════════════
def draw_panel(ax, iou, score, xlabel, rho_symbol='x',
               dot_size=6.0, alpha=0.85, density_method='kde',
               density_gamma=0.6, show_ylabel=True, seed=0):
    """在 ax 上画一幅密度散点图，并在左上角标注 Spearman ρ。"""
    rho = spearman(score, iou)

    dens = point_density(iou, score, method=density_method, seed=seed)
    # 密度小的点先画，密度大的点覆盖在上层，核心区才会呈现深色
    order = np.argsort(dens)
    vmax = np.percentile(dens, 99.0) if len(dens) > 100 else dens.max()

    ax.scatter(iou[order], score[order],
               c=dens[order], cmap=blue_cmap(),
               norm=PowerNorm(gamma=density_gamma, vmin=0.0, vmax=vmax),
               s=dot_size, alpha=alpha, linewidths=0,
               edgecolors='none', rasterized=True)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f'{v:.1f}' for v in np.arange(0, 1.01, 0.2)])
    ax.set_yticklabels([f'{v:.1f}' for v in np.arange(0, 1.01, 0.2)])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel, labelpad=6)
    if show_ylabel:
        ax.set_ylabel(YLABEL, labelpad=4)
    for side in ('top', 'right'):
        ax.spines[side].set_linewidth(0.8)

    ax.text(0.035, 0.965,
            rf'Spearman $\rho(s,{rho_symbol})$={rho:.2f}',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.32', facecolor='white',
                      edgecolor='0.35', linewidth=0.6))
    return rho


# ════════════════════════════════════════════════════════════════════════
#  单个面板（dots 样式 / CFC-Net 风格）
# ════════════════════════════════════════════════════════════════════════
def draw_panel_dots(ax, iou, score, xlabel, panel_letter=None, guide=None,
                    dot_size=2.0, alpha=0.9, color='blue', aspect=0.8,
                    annotate_rho=False, rho_symbol='x', show_ylabel=True):
    """纯蓝小散点 + 红色参考线，刻度 0.1 一档。

    guide: 'diag'  在 (a) 上画红色对角线 y=x；
           'vline' 在 (b) 上画 x=0.5 的红色虚线；
           None    不画参考线。
    panel_letter: 例如 '(a)'，画在横轴标签下方居中。
    annotate_rho: 参考图本身不标相关系数，默认关闭；打开则在左上角标注。
    """
    ax.scatter(iou, score, s=dot_size, c=color, alpha=alpha,
               linewidths=0, edgecolors='none', marker='o', rasterized=True)

    ticks = np.arange(0, 1.01, 0.2)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'{v:.1f}' for v in ticks])
    ax.set_yticklabels([f'{v:.1f}' for v in ticks])
    # 固定坐标轴框的长宽比，不受 figsize 影响（参考图约 0.8）
    ax.set_aspect(aspect, adjustable='box')
    ax.set_xlabel(xlabel, labelpad=3)
    if show_ylabel:
        ax.set_ylabel(YLABEL_CONF, labelpad=3)

    if panel_letter:
        # 子图编号单独放在横轴标签下方，与参考图一致
        ax.text(0.5, -0.22, panel_letter, transform=ax.transAxes,
                ha='center', va='top')

    rho = spearman(score, iou)
    if annotate_rho:
        ax.text(0.035, 0.965, rf'Spearman $\rho(s,{rho_symbol})$={rho:.2f}',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='square,pad=0.32', facecolor='white',
                          edgecolor='0.35', linewidth=0.6))
    return rho


# ════════════════════════════════════════════════════════════════════════
#  预处理 + 主绘图入口
# ════════════════════════════════════════════════════════════════════════
def prepare(cls_scores, iou_in, iou_out, max_dots=20000,
            drop_degenerate=False, cls_eps=0.02, iou_eps=0.02, seed=0):
    """过滤 + 下采样。

    drop_degenerate=True 时剔除退化样本：分类分数≈0（被判为背景的候选）
    以及 IoU≈0（完全没配上 GT 的候选）。这两类点在图上表现为 y=0 / x=0
    处的实心色带，会盖住真正想展示的相关性趋势。是否剔除请在论文里
    如实说明（例如「统计与某个 GT 有重叠的前景候选」）。
    """
    cls_scores = np.asarray(cls_scores, np.float64).ravel()
    iou_in = np.asarray(iou_in, np.float64).ravel()
    iou_out = np.asarray(iou_out, np.float64).ravel()

    keep = np.isfinite(cls_scores) & np.isfinite(iou_in) & np.isfinite(iou_out)
    if drop_degenerate:
        keep &= cls_scores > cls_eps
        keep &= iou_in > iou_eps
        keep &= iou_out > iou_eps
    cls_scores, iou_in, iou_out = cls_scores[keep], iou_in[keep], iou_out[keep]
    if len(cls_scores) == 0:
        raise RuntimeError('过滤后没有样本了，请调小 --cls-eps / 关闭 --drop-degenerate')

    if len(cls_scores) > max_dots:
        idx = np.random.default_rng(seed).choice(
            len(cls_scores), size=max_dots, replace=False)
        cls_scores, iou_in, iou_out = cls_scores[idx], iou_in[idx], iou_out[idx]
    return cls_scores, iou_in, iou_out


def plot_pair(cls_scores, iou_in, iou_out, out_dir, tag='',
              style='density',
              xlabel_in=None, xlabel_out=None,
              panel_size=3.2, dot_size=None, alpha=None,
              density_method='kde', density_gamma=0.6,
              dot_color='blue', annotate_rho=False, aspect=0.8,
              max_dots=20000, drop_degenerate=False,
              cls_eps=0.02, iou_eps=0.02, save_pdf=True, seed=0,
              font_file=None, dpi=600):
    """画左右两幅（回归前 / 回归后）相关性散点图。

    style='density' —— 密度着色 + 方形坐标系 + 左上角标注 Spearman ρ。
    style='dots'    —— CFC-Net 风格：纯蓝小点、0.1 一档刻度、
                       (a) 红色对角线 / (b) x=0.5 红色虚线、
                       子图编号单独放在底部、默认不标相关系数。

    同时输出：
      * 合并图 cls_iou_corr[_tag].png/.pdf
      * 两幅单独面板 cls_iou_corr[_tag]_anchor / _pred，便于用
        LaTeX subfigure 自己排版
    """
    os.makedirs(out_dir, exist_ok=True)
    apply_paper_style(font_file=font_file)

    s, x_in, x_out = prepare(cls_scores, iou_in, iou_out,
                             max_dots=max_dots,
                             drop_degenerate=drop_degenerate,
                             cls_eps=cls_eps, iou_eps=iou_eps, seed=seed)

    suffix = f'_{tag}' if tag else ''
    is_dots = style == 'dots'

    if is_dots:
        xlabel_in = xlabel_in or XLABEL_IN_PLAIN
        xlabel_out = xlabel_out or XLABEL_OUT_PLAIN
        panel_kw = dict(dot_size=2.0 if dot_size is None else dot_size,
                        alpha=0.9 if alpha is None else alpha,
                        color=dot_color, aspect=aspect,
                        annotate_rho=annotate_rho)
        draw = draw_panel_dots
        # 高度留够放 xlabel + 子图编号，实际框形由 set_aspect 决定
        fig_h, extra = panel_size * aspect + 0.95, 0.9
    else:
        xlabel_in = xlabel_in or XLABEL_IN
        xlabel_out = xlabel_out or XLABEL_OUT
        panel_kw = dict(dot_size=6.0 if dot_size is None else dot_size,
                        alpha=0.85 if alpha is None else alpha,
                        density_method=density_method,
                        density_gamma=density_gamma, seed=seed)
        draw = draw_panel
        fig_h, extra = panel_size, 0.6

    panels = [
        ('anchor', x_in, xlabel_in, 'x', '(a)', 'diag'),
        ('pred', x_out, xlabel_out, 'q', '(b)', 'vline'),
    ]

    def _kw(letter, guide):
        if is_dots:
            return dict(panel_kw, panel_letter=letter, guide=guide)
        return panel_kw

    # ── 合并图 ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(panel_size * 2 + extra, fig_h))
    rhos = {}
    for ax, (name, xv, xlab, sym, letter, guide) in zip(axes, panels):
        rhos[name] = draw(ax, xv, s, xlab, rho_symbol=sym,
                          **_kw(letter, guide))
    fig.tight_layout(w_pad=2.0)
    combined = osp.join(out_dir, f'cls_iou_corr{suffix}.png')
    save_figure(fig, combined, dpi=dpi, save_pdf=save_pdf)
    plt.close(fig)

    # ── 单独面板 ────────────────────────────────────────────────────
    for name, xv, xlab, sym, letter, guide in panels:
        fig, ax = plt.subplots(figsize=(panel_size, fig_h))
        draw(ax, xv, s, xlab, rho_symbol=sym, **_kw(letter, guide))
        fig.tight_layout()
        path = osp.join(out_dir, f'cls_iou_corr{suffix}_{name}.png')
        save_figure(fig, path, dpi=dpi, save_pdf=save_pdf)
        plt.close(fig)

    print(f'\n[✓] 已保存: {combined}（及 _anchor / _pred 单独面板、.pdf 版本）')
    print(f'    绘图样本数 N = {len(s):,}')
    print(f'    回归前  Spearman={rhos["anchor"]:.3f}  '
          f'Pearson={pearson(s, x_in):.3f}')
    print(f'    回归后  Spearman={rhos["pred"]:.3f}  '
          f'Pearson={pearson(s, x_out):.3f}')
    return rhos


# ════════════════════════════════════════════════════════════════════════
#  命令行：从 .npz 重新出图
# ════════════════════════════════════════════════════════════════════════
def add_style_args(parser):
    """把样式相关参数挂到任意 parser 上（收集脚本也复用这些参数）。"""
    parser.add_argument('--plot-style', default='density',
                        choices=['density', 'dots'],
                        help='density=密度着色+标注ρ；dots=CFC-Net 风格纯蓝小点')
    parser.add_argument('--dot-color', default='blue',
                        help='dots 样式的散点颜色')
    parser.add_argument('--annotate-rho', action='store_true',
                        help='dots 样式下也在左上角标注 Spearman ρ（默认不标）')
    parser.add_argument('--aspect', type=float, default=0.8,
                        help='dots 样式坐标轴框的高/宽比（参考图约 0.8）')
    parser.add_argument('--max-dots', type=int, default=20000,
                        help='参与绘图的散点数上限（下采样）')
    parser.add_argument('--drop-degenerate', action='store_true',
                        help='剔除分类分数≈0 / IoU≈0 的退化样本（去掉边缘色带）')
    parser.add_argument('--cls-eps', type=float, default=0.02)
    parser.add_argument('--iou-eps', type=float, default=0.02)
    parser.add_argument('--density-method', default='kde',
                        choices=['kde', 'hist'],
                        help='kde 更平滑（需 scipy），hist 更快')
    parser.add_argument('--density-gamma', type=float, default=0.6,
                        help='密度→颜色的幂次，越小浅色区越明显')
    parser.add_argument('--dot-size', type=float, default=None,
                        help='默认按样式取值：density=6.0，dots=2.0')
    parser.add_argument('--alpha', type=float, default=None,
                        help='默认按样式取值：density=0.85，dots=0.9')
    parser.add_argument('--panel-size', type=float, default=3.2,
                        help='单个面板宽度（英寸）')
    parser.add_argument('--xlabel-in', default=None)
    parser.add_argument('--xlabel-out', default=None)
    parser.add_argument('--font-file', default=None,
                        help='Times New Roman 的 .ttf 路径（容器里没装字体时用）')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--seed', type=int, default=0)
    return parser


def style_kwargs(args):
    return dict(style=args.plot_style,
                dot_color=args.dot_color,
                annotate_rho=args.annotate_rho,
                aspect=args.aspect,
                max_dots=args.max_dots,
                drop_degenerate=args.drop_degenerate,
                cls_eps=args.cls_eps, iou_eps=args.iou_eps,
                density_method=args.density_method,
                density_gamma=args.density_gamma,
                dot_size=args.dot_size, alpha=args.alpha,
                panel_size=args.panel_size,
                xlabel_in=args.xlabel_in, xlabel_out=args.xlabel_out,
                font_file=args.font_file, dpi=args.dpi,
                seed=args.seed)


def main():
    parser = argparse.ArgumentParser(
        description='从缓存的 .npz 重新绘制分类分数-IoU 相关性密度散点图')
    parser.add_argument('npz', help='vis_cls_reg_correlation.py 导出的 .npz')
    parser.add_argument('--out-dir', default='work_vis/cls_reg_correlation')
    parser.add_argument('--tag', default='')
    add_style_args(parser)
    args = parser.parse_args()

    data = np.load(args.npz)
    plot_pair(data['cls_scores'], data['iou_in'], data['iou_out'],
              args.out_dir, tag=args.tag, **style_kwargs(args))


if __name__ == '__main__':
    main()
