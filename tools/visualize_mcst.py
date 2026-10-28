"""

python tools/visualize_mcst.py \
    projects/mcst/dior_mcst-s_fpn_oriented_1x.py \
    work_dirs/DIOR/dior_mcst-s_fpn_oriented_fix/best_dota_mAP_epoch_12.pth \
    --images demo/demo.png \
    --out work_vis/vis/demo.png

"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from mmdet.apis import inference_detector, init_detector  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402

import mmrotate  # noqa: F401,E402  (registers MCSTNet and the rotated heads)
from mmrotate.models.backbones.mcstnet import MCSTblock  # noqa: E402

COLUMN_TITLES = ('Input', 'STM  $\\tilde{X}$', 'FEM  $\\tilde{Y}$',
                 'Gate  $m$', '$1-m$', 'Detection  $\\tilde{U}$')


class MCSTFeatureCatcher:
    """Capture X̃, Ỹ, m and Ũ from one MCSTblock, plus the backbone outputs."""

    def __init__(self, backbone, stage: int, block_idx: int):
        blocks = getattr(backbone, f'block{stage}')
        if block_idx >= len(blocks):
            raise ValueError(
                f'stage {stage} only has {len(blocks)} blocks, '
                f'got --block-idx {block_idx}')
        block = blocks[block_idx].attn.spatial_gating_unit
        if not isinstance(block, MCSTblock):
            raise TypeError(
                f'expected an MCSTblock, got {type(block).__name__}. '
                'Is this config really using the MCSTNet backbone?')

        self.feats = {}
        self.handles = [
            block.st.register_forward_hook(self._store('x_tilde')),
            block.fg.register_forward_hook(self._store('y_tilde')),
            block.local_att.register_forward_hook(self._store('xl')),
            block.global_att.register_forward_hook(self._store('xg')),
            block.register_forward_hook(self._store('u_tilde')),
            backbone.register_forward_hook(self._store_backbone),
        ]

    def _store(self, name):
        def hook(_module, _inputs, output):
            self.feats[name] = output.detach()
        return hook

    def _store_backbone(self, _module, _inputs, output):
        self.feats['backbone_outs'] = [o.detach() for o in output]

    @property
    def gate(self) -> torch.Tensor:
        """m = sigmoid(local_att(z) + global_att(z)), broadcasting the global path."""
        xl, xg = self.feats['xl'], self.feats['xg']
        if xg.shape[2:] != (1, 1) and xg.shape[2:] != xl.shape[2:]:
            xg = torch.nn.functional.interpolate(
                xg, size=xl.shape[2:], mode='bilinear', align_corners=False)
        return torch.sigmoid(xl + xg)

    def clear(self):
        self.feats.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def to_heatmap(feat: torch.Tensor,
               out_size,
               reduce: str = 'mean',
               normalize: bool = True) -> np.ndarray:
    """Collapse a (1, C, H, W) tensor to a HxW map resized to ``out_size``."""
    fmap = feat[0].float()
    if reduce == 'mean':
        fmap = fmap.mean(0)
    elif reduce == 'l2':
        fmap = fmap.pow(2).mean(0).sqrt()
    elif reduce == 'max':
        fmap = fmap.max(0)[0]
    else:
        raise ValueError(f'unknown reduce mode: {reduce}')

    fmap = fmap.cpu().numpy()
    fmap = cv2.resize(fmap, out_size, interpolation=cv2.INTER_CUBIC)
    if normalize:
        lo, hi = np.percentile(fmap, 1), np.percentile(fmap, 99)
        fmap = np.clip((fmap - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return fmap


def draw_boxes(image: np.ndarray, result, score_thr: float) -> np.ndarray:
    """Draw predicted rotated boxes (cx, cy, w, h, angle) in red."""
    canvas = image.copy()
    pred = result.pred_instances
    keep = pred.scores.cpu().numpy() >= score_thr
    # depending on boxtype2tensor this is either a RotatedBoxes or a raw Tensor
    boxes = getattr(pred.bboxes, 'tensor', pred.bboxes)
    boxes = boxes.cpu().numpy()[keep]
    for cx, cy, w, h, angle in boxes:
        pts = cv2.boxPoints(((cx, cy), (w, h), np.degrees(angle)))
        cv2.polylines(canvas, [np.int32(pts)], True, (255, 0, 0), 2, cv2.LINE_AA)
    return canvas


def visualize(args):
    init_default_scope('mmrotate')
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()

    catcher = MCSTFeatureCatcher(model.backbone, args.stage, args.block_idx)

    rows = []
    for img_path in args.images:
        catcher.clear()
        with torch.no_grad():
            result = inference_detector(model, img_path)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        size = (image.shape[1], image.shape[0])

        # m usually sits in a narrow band around 0.5, so stretch it by default;
        # --raw-gate keeps the absolute (0, 1) values instead.
        gate_map = to_heatmap(
            catcher.gate, size, 'mean', normalize=not args.raw_gate)
        rows.append([
            draw_boxes(image, result, args.score_thr),
            to_heatmap(catcher.feats['x_tilde'], size, args.reduce),
            to_heatmap(catcher.feats['y_tilde'], size, args.reduce),
            gate_map,
            1.0 - gate_map,
            to_heatmap(catcher.feats['backbone_outs'][args.det_stage - 1],
                       size, args.reduce),
        ])
    catcher.remove()

    n_rows, n_cols = len(rows), len(COLUMN_TITLES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(args.col_width * n_cols, args.col_width * n_rows),
        squeeze=False)

    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            ax = axes[r][c]
            if c == 0:
                ax.imshow(cell)
            else:
                ax.imshow(cell, cmap=args.cmap, vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0 and not args.no_titles:
                ax.set_title(COLUMN_TITLES[c], fontsize=12)

    plt.tight_layout(pad=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {n_rows} x {n_cols} figure -> {out}')


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize MCSTNet features')
    parser.add_argument('config', help='MCST config file')
    parser.add_argument('checkpoint', help='Trained detector checkpoint')
    parser.add_argument(
        '--images', nargs='+', required=True, help='Input image paths')
    parser.add_argument(
        '--out', default='work_dirs/vis/mcst_features.png', help='Output figure')
    parser.add_argument(
        '--stage', type=int, default=2, choices=(1, 2, 3, 4),
        help='Which backbone stage supplies columns 2-5 (default: 2)')
    parser.add_argument(
        '--block-idx', type=int, default=0,
        help='Which block inside that stage (default: 0)')
    parser.add_argument(
        '--det-stage', type=int, default=4, choices=(1, 2, 3, 4),
        help='Backbone stage used for the last column (default: 4)')
    parser.add_argument(
        '--reduce', default='mean', choices=('mean', 'l2', 'max'),
        help='How to collapse channels into a 2D map (default: mean)')
    parser.add_argument(
        '--raw-gate', action='store_true',
        help='Show absolute (0,1) gate values instead of stretching contrast')
    parser.add_argument('--cmap', default='jet', help='Matplotlib colormap')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--col-width', type=float, default=3.0)
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--no-titles', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
