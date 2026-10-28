"""把旋转框建模成 2D 高斯分布，画成分层色带热力图叠加到原图上。

高斯参数取 GWD/KLD 的常用约定：均值为框中心，
协方差 ``Σ = R diag((sw/2)^2, (sh/2)^2) R^T``（``s`` 是 --sigma-scale），
即框的内接椭圆恰好对应 1σ 的等值线。图上每条色带边界是等 Mahalanobis
半径的椭圆，所以色带宽度看起来是均匀的（论文里常见的样子）。

用法一：直接读 DIOR 的 OBB 标注，只画第 0 个 airplane 并裁剪到目标附近::

    python tools/analysis_tools/plot_obb_gaussian.py \
        --img ../datasets/obb/DIOR/JPEGImages-trainval/00491.jpg \
        --dior-ann '../datasets/obb/DIOR/Annotations/Oriented Bounding Boxes/00491.xml' \
        --classes airplane --keep 0 --crop \
        --out work_vis/obb_gaussian/airplane_gaussian.png

用法二：手动给旋转框（角度默认 deg，正值在图像坐标里顺时针；
le90 的弧度加 --angle-unit rad），可重复传 --obb 画多个::

    python tools/analysis_tools/plot_obb_gaussian.py \
        --img demo/demo.png --obb 400 400 320 110 -30 \
        --draw-box --colorbar --out work_vis/obb_gaussian/demo.png

用法三：读 DOTA 格式 txt（8 点多边形 + 类别 + difficult），换连续热力图::

    python tools/analysis_tools/plot_obb_gaussian.py \
        --img P0001.png --dota-ann P0001.txt --mode smooth --crop
"""

import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MIN_SIDE = 1.0  # DIOR 有高为 0 的退化标注，协方差会奇异，这里兜底

_SERIF_CANDIDATES = [
    'Times New Roman', 'Nimbus Roman', 'Nimbus Roman No9 L',
    'Liberation Serif', 'Tinos', 'FreeSerif', 'DejaVu Serif',
]


def apply_paper_style(font_size=13.0):
    available = {f.name for f in fm.fontManager.ttflist}
    font = next((n for n in _SERIF_CANDIDATES if n in available), 'DejaVu Serif')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [font] + _SERIF_CANDIDATES,
        'mathtext.fontset': 'stix',
        'font.size': font_size,
        'axes.unicode_minus': False,
    })


# ------------------------------ 旋转框读取 ------------------------------


def poly_to_obb(poly):
    """8 个数的多边形 -> (cx, cy, w, h, theta)，theta 为弧度。"""
    pts = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    return np.array([cx, cy, w, h, np.deg2rad(angle)], dtype=np.float64)


def obb_to_poly(obb):
    """(cx, cy, w, h, theta) -> (4, 2) 角点，用于画框轮廓。"""
    cx, cy, w, h, theta = obb
    return cv2.boxPoints(((cx, cy), (w, h), np.rad2deg(theta)))


def load_dior_xml(path, classes=None):
    """读 DIOR 的 Oriented Bounding Boxes xml，返回 [(obb, name), ...]。"""
    keys = ('x_left_top', 'y_left_top', 'x_right_top', 'y_right_top',
            'x_right_bottom', 'y_right_bottom', 'x_left_bottom',
            'y_left_bottom')
    out = []
    for obj in ET.parse(path).getroot().findall('object'):
        name = obj.findtext('name', default='').strip()
        if classes and name not in classes:
            continue
        node = obj.find('robndbox')
        if node is None:  # 少数条目只有水平框
            continue
        poly = [float(node.findtext(k)) for k in keys]
        out.append((poly_to_obb(poly), name))
    return out


def load_dota_txt(path, classes=None):
    """读 DOTA 格式 txt：x1 y1 ... x4 y4 class difficult。"""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9 or not parts[0].replace('.', '', 1).lstrip(
                    '-').isdigit():
                continue  # 跳过 imagesource/gsd 等表头行
            name = parts[8]
            if classes and name not in classes:
                continue
            out.append((poly_to_obb([float(v) for v in parts[:8]]), name))
    return out


def collect_obbs(args):
    """汇总所有来源的旋转框，返回 [(obb, name), ...]。"""
    obbs = []
    for values in args.obb or []:
        cx, cy, w, h, angle = values
        theta = angle if args.angle_unit == 'rad' else np.deg2rad(angle)
        obbs.append((np.array([cx, cy, w, h, theta], dtype=np.float64), ''))
    for values in args.poly or []:
        obbs.append((poly_to_obb(values), ''))
    if args.dior_ann:
        obbs += load_dior_xml(args.dior_ann, args.classes)
    if args.dota_ann:
        obbs += load_dota_txt(args.dota_ann, args.classes)

    obbs = [(o, n) for o, n in obbs if min(o[2], o[3]) >= MIN_SIDE]
    if args.keep is not None:
        if not 0 <= args.keep < len(obbs):
            raise ValueError(f'--keep {args.keep} 越界，可用目标数 {len(obbs)}')
        obbs = [obbs[args.keep]]
    return obbs


# ------------------------------ 高斯与网格 ------------------------------


def obb_to_gaussian(obb, sigma_scale=1.0):
    """(cx, cy, w, h, theta) -> 均值 (2,) 与协方差 (2, 2)。"""
    cx, cy, w, h, theta = obb
    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.array([[cos, -sin], [sin, cos]])  # 图像坐标 y 向下 -> 视觉上顺时针
    scale = np.diag([(w * sigma_scale / 2) ** 2, (h * sigma_scale / 2) ** 2])
    return np.array([cx, cy]), rot @ scale @ rot.T


def mahalanobis_field(mu, cov, xx, yy):
    """网格上每点到高斯中心的 Mahalanobis 半径 r（r=1 即 1σ 椭圆）。"""
    inv = np.linalg.inv(cov)
    dx, dy = xx - mu[0], yy - mu[1]
    r2 = (inv[0, 0] * dx * dx + (inv[0, 1] + inv[1, 0]) * dx * dy
          + inv[1, 1] * dy * dy)
    return np.sqrt(np.maximum(r2, 0.0))


def crop_window(obbs, img_shape, margin, reach):
    """裁剪窗口 (x0, y0, x1, y1)。

    ``reach = r_max * sigma_scale`` 是最外层等值线相对原框的放大倍数，
    先把框按 reach 放大以框住整个热力图，再按 margin 留一点余量。
    """
    h_img, w_img = img_shape[:2]
    pts = np.concatenate(
        [obb_to_poly([o[0], o[1], o[2] * reach, o[3] * reach, o[4]])
         for o, _ in obbs], axis=0)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    pad_x, pad_y = margin * (x1 - x0), margin * (y1 - y0)
    return (int(max(0, np.floor(x0 - pad_x))),
            int(max(0, np.floor(y0 - pad_y))),
            int(min(w_img, np.ceil(x1 + pad_x))),
            int(min(h_img, np.ceil(y1 + pad_y))))


def band_colors(cmap, n, lo, hi):
    """从 cmap 的 [lo, hi] 区间均匀取 n 个颜色，中心暖色、外圈冷色。"""
    return plt.get_cmap(cmap)(np.linspace(lo, hi, n))[::-1]


# ------------------------------ 绘图 ------------------------------


def draw_bands(ax, mu, cov, shape, args, colors):
    """分层色带：等 Mahalanobis 半径的填充等值线（论文常见样式）。"""
    h, w = shape
    step = max(1, args.grid_step)
    ys, xs = np.arange(0, h, step), np.arange(0, w, step)
    xx, yy = np.meshgrid(xs, ys)
    field = mahalanobis_field(mu, cov, xx, yy)

    levels = np.linspace(0.0, args.r_max, args.levels + 1)
    cset = ax.contourf(xs, ys, field, levels=levels, colors=colors,
                       alpha=args.alpha, antialiased=True)
    # 去掉相邻色带间的白缝；mpl>=3.8 的 ContourSet 本身就是 Collection
    for coll in getattr(cset, 'collections', [cset]):
        coll.set_edgecolor('none')
    if args.edge_lw > 0:
        ax.contour(xs, ys, field, levels=levels[1:], colors='white',
                   linewidths=args.edge_lw, alpha=min(1.0, args.alpha + 0.2))
    return cset


def density_overlay(mus_covs, shape, args):
    """连续高斯热力图：多目标取 max，alpha 随密度衰减。"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    dens = np.zeros((h, w), dtype=np.float64)
    for mu, cov in mus_covs:
        r = mahalanobis_field(mu, cov, xx, yy)
        dens = np.maximum(dens, np.exp(-0.5 * np.square(r)))

    cmap = LinearSegmentedColormap.from_list(
        'trunc', band_colors(args.cmap, 256, *args.cmap_range)[::-1])
    rgba = cmap(dens)
    floor = np.exp(-0.5 * args.r_max ** 2)  # r_max 之外直接透明
    fade = np.clip((dens - floor) / (1.0 - floor), 0.0, 1.0)
    rgba[..., 3] = args.alpha * fade ** args.gamma
    return rgba


def plot(args):
    bgr = cv2.imread(args.img)
    if bgr is None:
        raise FileNotFoundError(f'读不到图像：{args.img}')
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    obbs = collect_obbs(args)
    if not obbs:
        raise ValueError('没有可用的旋转框，请检查 --obb/--poly/--dior-ann/--classes')
    print(f'[i] 待绘制目标 {len(obbs)} 个：'
          f'{[n or "obb" for _, n in obbs]}')

    if args.crop:
        x0, y0, x1, y1 = crop_window(obbs, image.shape, args.margin,
                                     args.r_max * args.sigma_scale)
        image = image[y0:y1, x0:x1]
        obbs = [(np.array([o[0] - x0, o[1] - y0, o[2], o[3], o[4]]), n)
                for o, n in obbs]
        print(f'[i] 裁剪窗口 x[{x0}, {x1}) y[{y0}, {y1})')

    mus_covs = [obb_to_gaussian(o, args.sigma_scale) for o, _ in obbs]

    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(args.fig_width,
                                    args.fig_width * h / max(w, 1)))
    ax.imshow(image, interpolation='bilinear')

    colors = band_colors(args.cmap, args.levels, *args.cmap_range)
    cset = None
    if args.mode == 'band':
        for mu, cov in mus_covs:
            cset = draw_bands(ax, mu, cov, (h, w), args, colors)
    else:
        ax.imshow(density_overlay(mus_covs, (h, w), args),
                  interpolation='bilinear')

    if args.draw_box:
        for obb, _ in obbs:
            pts = obb_to_poly(obb)
            ax.plot(*np.append(pts, pts[:1], axis=0).T, color=args.box_color,
                    lw=args.box_lw, ls='--')

    if args.colorbar and cset is not None:
        cbar = fig.colorbar(cset, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(r'Mahalanobis radius  $r$  ($\sigma$)')
        cbar.solids.set_alpha(1.0)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    if args.title:
        ax.set_title(args.title, pad=8)
    fig.tight_layout(pad=0.1)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description='旋转框 -> 2D 高斯分层热力图可视化')
    parser.add_argument('--img', required=True, help='底图路径')
    parser.add_argument('--obb', type=float, nargs=5, action='append',
                        metavar=('CX', 'CY', 'W', 'H', 'ANGLE'),
                        help='手动指定旋转框，可重复传多个')
    parser.add_argument('--poly', type=float, nargs=8, action='append',
                        metavar='V', help='手动指定 4 点多边形，可重复传多个')
    parser.add_argument('--dior-ann', default=None,
                        help='DIOR Oriented Bounding Boxes 的 xml 路径')
    parser.add_argument('--dota-ann', default=None, help='DOTA 格式 txt 路径')
    parser.add_argument('--classes', nargs='+', default=None,
                        help='只画这些类别（配合标注文件使用）')
    parser.add_argument('--keep', type=int, default=None,
                        help='只画第 k 个目标（0 起），默认全画')
    parser.add_argument('--angle-unit', choices=['deg', 'rad'], default='deg',
                        help='--obb 里角度的单位，le90 弧度传 rad')

    parser.add_argument('--mode', choices=['band', 'smooth'], default='band',
                        help='band=分层色带（默认），smooth=连续热力图')
    parser.add_argument('--sigma-scale', type=float, default=1.0,
                        help='σ 相对框半边长的缩放，越大高斯越胖')
    parser.add_argument('--r-max', type=float, default=1.5,
                        help='最外层等值线的 Mahalanobis 半径（单位 σ），'
                             '即热力图相对框放大的倍数')
    parser.add_argument('--levels', type=int, default=6, help='色带层数')
    parser.add_argument('--cmap', default='jet', help='配色，如 jet / turbo')
    parser.add_argument('--cmap-range', type=float, nargs=2,
                        default=(0.15, 0.95), metavar=('LO', 'HI'),
                        help='只取 cmap 的这段区间，避开 jet 两端的深色')
    parser.add_argument('--alpha', type=float, default=0.5, help='叠加透明度')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='smooth 模式下 alpha 的衰减指数')
    parser.add_argument('--grid-step', type=int, default=1,
                        help='band 模式的网格采样步长，大图可设 2')
    parser.add_argument('--edge-lw', type=float, default=0.0,
                        help='色带之间的白色分界线宽，0 表示不画')

    parser.add_argument('--crop', action='store_true', help='裁剪到目标附近')
    parser.add_argument('--margin', type=float, default=0.12,
                        help='裁剪时在热力图范围外再留的比例')
    parser.add_argument('--draw-box', action='store_true', help='叠加框轮廓')
    parser.add_argument('--box-color', default='white')
    parser.add_argument('--box-lw', type=float, default=2.0)
    parser.add_argument('--colorbar', action='store_true')
    parser.add_argument('--title', default=None)

    parser.add_argument('--out', default='work_vis/obb_gaussian/obb_gaussian.png',
                        help='输出图片路径（同名 pdf 一并保存）')
    parser.add_argument('--fig-width', type=float, default=5.0)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--font-size', type=float, default=12.0)
    return parser.parse_args()


def main():
    args = parse_args()
    apply_paper_style(args.font_size)
    fig = plot(args)

    out = args.out
    os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
    stem, _ = osp.splitext(out)
    for path in (out, f'{stem}.pdf'):
        fig.savefig(path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.02)
        print(f'[✓] 已保存 {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
