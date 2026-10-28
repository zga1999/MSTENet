"""
python tools/visualize_vit.py \
    projects/vit/dior_vit_fpn_orientedstdcbbn_dualfocal_1x.py \
    work_dirs/DIOR/dior_hivit_fpn_oriented/best_dota_mAP_epoch_12.pth \
    --images demo/P2781__1__824___0.png \
    --out work_vis/vis/vit_P2781.png

    work_dirs/DIOR/dior_hivit_fpn_oriented/best_dota_mAP_epoch_12.pth \

    work_dirs/DIOR/dior_vit_fpn_orientedstdcbbn_dualfocal_fix/best_dota_mAP_epoch_11.pth \
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

import mmrotate  # noqa: F401,E402

COLUMN_TITLES = (
    'Input',
    'Attn early',
    'Attn mid',
    'Attn late',
    'Attn rollout',
    'Feat overlay',
)


class ViTAttnCatcher:
    """Capture Attention matrices from selected blocks (+ late tokens)."""

    def __init__(self, backbone, attn_blocks, late_block: int):
        n = len(backbone.blocks)
        self.attn_blocks = list(attn_blocks)
        for idx in self.attn_blocks + [late_block]:
            if not 0 <= idx < n:
                raise ValueError(f'block index {idx} out of range [0, {n})')

        self.feats = {'attn': {}}
        self.handles = []
        for idx in self.attn_blocks:
            self.handles.append(
                backbone.blocks[idx].attn.register_forward_hook(
                    self._store_attn(idx)))
        self.handles.append(
            backbone.blocks[late_block].register_forward_hook(
                self._store_tokens('late')))

    def _store_attn(self, idx):
        def hook(_module, _inputs, output):
            # Attention.forward → (tokens, attn) with attn (B, heads, N, N)
            attn = output[1] if isinstance(output, tuple) else output
            self.feats['attn'][idx] = attn.detach()
        return hook

    def _store_tokens(self, name):
        def hook(_module, _inputs, output):
            tok = output[0] if isinstance(output, tuple) else output
            self.feats[name] = tok.detach()
        return hook

    def clear(self):
        self.feats = {'attn': {}}

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def cls_attention_map(attn: torch.Tensor, smooth: int = 0) -> np.ndarray:
    """(B, heads, N, N) → HxW CLS→patch map in [0, 1]."""
    # average heads; CLS query attends to patch keys (skip CLS column)
    cls_attn = attn[0, :, 0, 1:].mean(0).float().cpu().numpy()
    side = int(round(cls_attn.size ** 0.5))
    if side * side != cls_attn.size:
        raise RuntimeError(f'cannot reshape {cls_attn.size} to square')
    fmap = cls_attn.reshape(side, side)
    if smooth > 0:
        k = smooth if smooth % 2 == 1 else smooth + 1
        fmap = cv2.GaussianBlur(fmap, (k, k), 0)
    fmap = fmap - fmap.min()
    fmap = fmap / (fmap.max() + 1e-8)
    return fmap


def attention_rollout(attns: list, smooth: int = 0) -> np.ndarray:
    """Abnar & Zuidema attention rollout on CLS→patches.

    Each ``attn`` is (B, heads, N, N). We average heads, add residual, and
    left-multiply across layers.
    """
    result = None
    for attn in attns:
        a = attn[0].float().mean(0)  # (N, N)
        eye = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result
    # CLS row → patches
    cls = result[0, 1:].cpu().numpy()
    side = int(round(cls.size ** 0.5))
    fmap = cls.reshape(side, side)
    if smooth > 0:
        k = smooth if smooth % 2 == 1 else smooth + 1
        fmap = cv2.GaussianBlur(fmap, (k, k), 0)
    fmap = fmap - fmap.min()
    fmap = fmap / (fmap.max() + 1e-8)
    return fmap


def token_activation_map(tokens: torch.Tensor) -> np.ndarray:
    """(B, 1+HW, C) → HxW L2 activation in [0, 1] (drop CLS)."""
    tok = tokens[0, 1:].float()  # (HW, C)
    act = tok.norm(dim=-1).cpu().numpy()
    side = int(round(act.size ** 0.5))
    fmap = act.reshape(side, side)
    lo, hi = np.percentile(fmap, 1), np.percentile(fmap, 99)
    fmap = np.clip((fmap - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return fmap


def overlay_on_image(image: np.ndarray,
                     fmap: np.ndarray,
                     alpha: float = 0.45,
                     cmap: str = 'jet') -> np.ndarray:
    """Blend a [0,1] spatial map onto an RGB image (classic ViT look)."""
    h, w = image.shape[:2]
    heat = cv2.resize(fmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0.0, 1.0)
    colored = plt.get_cmap(cmap)(heat)[..., :3]  # RGBA → RGB float
    colored = (colored * 255.0).astype(np.float32)
    base = image.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * colored
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boxes(image: np.ndarray, result, score_thr: float) -> np.ndarray:
    canvas = image.copy()
    pred = result.pred_instances
    keep = pred.scores.cpu().numpy() >= score_thr
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
    if hasattr(model.backbone, 'use_checkpoint'):
        model.backbone.use_checkpoint = False

    attn_blocks = [args.early, args.mid, args.late]
    # rollout uses every block up to late for a fuller propagation
    rollout_blocks = list(range(args.late + 1)) if args.full_rollout else attn_blocks
    hook_blocks = sorted(set(attn_blocks) | set(rollout_blocks))
    catcher = ViTAttnCatcher(model.backbone, hook_blocks, args.late)

    rows = []
    for img_path in args.images:
        catcher.clear()
        with torch.no_grad():
            result = inference_detector(model, img_path)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        attns = catcher.feats['attn']

        early_map = cls_attention_map(attns[args.early], args.smooth)
        mid_map = cls_attention_map(attns[args.mid], args.smooth)
        late_map = cls_attention_map(attns[args.late], args.smooth)
        rollout_map = attention_rollout(
            [attns[i] for i in rollout_blocks], args.smooth)
        feat_map = token_activation_map(catcher.feats['late'])

        rows.append([
            draw_boxes(image, result, args.score_thr),
            overlay_on_image(image, early_map, args.alpha, args.cmap),
            overlay_on_image(image, mid_map, args.alpha, args.cmap),
            overlay_on_image(image, late_map, args.alpha, args.cmap),
            overlay_on_image(image, rollout_map, args.alpha, args.cmap),
            overlay_on_image(image, feat_map, args.alpha, args.cmap),
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
            ax.imshow(cell)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0 and not args.no_titles:
                title = COLUMN_TITLES[c]
                if c == 1:
                    title = f'Attn @{args.early}'
                elif c == 2:
                    title = f'Attn @{args.mid}'
                elif c == 3:
                    title = f'Attn @{args.late}'
                ax.set_title(title, fontsize=12)

    plt.tight_layout(pad=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {n_rows} x {n_cols} figure -> {out}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Classic ViT attention / feature overlays')
    parser.add_argument('config', help='ViT detector config')
    parser.add_argument('checkpoint', help='Trained detector checkpoint')
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--out', default='work_dirs/vis/vit_features.png')
    parser.add_argument('--early', type=int, default=3, help='Early attn block')
    parser.add_argument('--mid', type=int, default=7, help='Mid attn block')
    parser.add_argument('--late', type=int, default=11, help='Late attn / feat block')
    parser.add_argument(
        '--full-rollout', action='store_true',
        help='Rollout over blocks 0..late (default: only early/mid/late)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Heatmap blend weight on the RGB image')
    parser.add_argument(
        '--smooth', type=int, default=3,
        help='Odd Gaussian kernel for attention maps (0=off, default: 3)')
    parser.add_argument('--cmap', default='jet')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--col-width', type=float, default=3.0)
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--no-titles', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
