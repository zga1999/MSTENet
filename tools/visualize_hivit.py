"""
    python tools/visualize_hivit.py \
        projects/hivit/dior_hivit_fpn_oriented_1x.py \
        work_dirs/DIOR/dior_hivit_fpn_oriented/best_dota_mAP_epoch_12.pth \
        --images demo/P2781__1__824___0.png \
        --out work_vis/vis/hivit_P2781.png \
        --full-rollout --smooth 5 --alpha 0.5
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


class HiViTAttnCatcher:
    """Capture full-map Attention from selected main blocks (+ late tokens)."""

    def __init__(self, backbone, attn_blocks, late_block: int):
        self.main = backbone.blocks[-backbone.num_main_blocks:]
        n = len(self.main)
        self.attn_blocks = list(attn_blocks)
        for idx in self.attn_blocks + [late_block]:
            if not 0 <= idx < n:
                raise ValueError(f'main block {idx} out of range [0, {n})')
            if self.main[idx].attn is None:
                raise TypeError(
                    f'main block {idx} has no Attention; '
                    'pick a global_indices block (e.g. 4/9/14/19)')

        self.patch_size = backbone.patch_size
        self.feats = {'attn': {}}
        self._hw = None
        self.handles = [
            backbone.register_forward_pre_hook(self._store_input_hw),
        ]
        for idx in self.attn_blocks:
            # Attention.forward returns tokens only; attn_drop sees (B,H,N,N)
            self.handles.append(
                self.main[idx].attn.attn_drop.register_forward_hook(
                    self._store_attn(idx)))
        self.handles.append(
            self.main[late_block].register_forward_hook(
                self._store_tokens('late')))

    def _store_input_hw(self, _module, inputs):
        self._hw = (int(inputs[0].shape[2]), int(inputs[0].shape[3]))

    def _store_attn(self, idx):
        def hook(_module, _inputs, output):
            self.feats['attn'][idx] = output.detach()
        return hook

    def _store_tokens(self, name):
        def hook(_module, _inputs, output):
            tok = output[0] if isinstance(output, tuple) else output
            self.feats[name] = tok.detach()
        return hook

    def clear(self):
        self.feats = {'attn': {}}
        self._hw = None

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _to_spatial(vec: np.ndarray, hw, patch_size) -> np.ndarray:
    """Flattened length-N map → HxW using image patch grid when possible."""
    hp = hw[0] // patch_size
    wp = hw[1] // patch_size
    if vec.size == hp * wp:
        return vec.reshape(hp, wp)
    side = int(round(vec.size ** 0.5))
    if side * side != vec.size:
        raise RuntimeError(
            f'cannot reshape attn length {vec.size} to spatial map '
            f'(expected {hp}x{wp} for global attention)')
    return vec.reshape(side, side)


def mean_attention_map(attn: torch.Tensor,
                       hw,
                       patch_size,
                       smooth: int = 0) -> np.ndarray:
    """(B, heads, N, N) → HxW mean query→key map in [0, 1] (no CLS)."""
    # average heads + queries → how much each patch is attended to
    key_score = attn[0].float().mean(dim=(0, 1)).cpu().numpy()
    fmap = _to_spatial(key_score, hw, patch_size)
    if smooth > 0:
        k = smooth if smooth % 2 == 1 else smooth + 1
        fmap = cv2.GaussianBlur(fmap, (k, k), 0)
    fmap = fmap - fmap.min()
    return fmap / (fmap.max() + 1e-8)


def attention_rollout(attns: list,
                      hw,
                      patch_size,
                      smooth: int = 0) -> np.ndarray:
    """Abnar & Zuidema attention rollout (CLS-free: mean over queries)."""
    result = None
    for attn in attns:
        a = attn[0].float().mean(0)  # (N, N)
        eye = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result
    score = result.mean(0).cpu().numpy()
    fmap = _to_spatial(score, hw, patch_size)
    if smooth > 0:
        k = smooth if smooth % 2 == 1 else smooth + 1
        fmap = cv2.GaussianBlur(fmap, (k, k), 0)
    fmap = fmap - fmap.min()
    return fmap / (fmap.max() + 1e-8)


def token_activation_map(tokens: torch.Tensor, hw, patch_size) -> np.ndarray:
    """(B, HW, C) → HxW L2 activation in [0, 1]."""
    act = tokens[0].float().norm(dim=-1).cpu().numpy()
    fmap = _to_spatial(act, hw, patch_size)
    lo, hi = np.percentile(fmap, 1), np.percentile(fmap, 99)
    return np.clip((fmap - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def overlay_on_image(image: np.ndarray,
                     fmap: np.ndarray,
                     alpha: float = 0.45,
                     cmap: str = 'jet') -> np.ndarray:
    h, w = image.shape[:2]
    heat = cv2.resize(fmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0.0, 1.0)
    colored = plt.get_cmap(cmap)(heat)[..., :3]
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
    globals_ = list(getattr(model.backbone, 'global_indices', attn_blocks))
    if args.full_rollout:
        rollout_blocks = [i for i in globals_ if i <= args.late]
        if not rollout_blocks:
            rollout_blocks = attn_blocks
    else:
        rollout_blocks = attn_blocks
    hook_blocks = sorted(set(attn_blocks) | set(rollout_blocks))
    catcher = HiViTAttnCatcher(model.backbone, hook_blocks, args.late)
    ps = catcher.patch_size

    rows = []
    for img_path in args.images:
        catcher.clear()
        with torch.no_grad():
            result = inference_detector(model, img_path)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        attns = catcher.feats['attn']
        hw = catcher._hw

        early_map = mean_attention_map(attns[args.early], hw, ps, args.smooth)
        mid_map = mean_attention_map(attns[args.mid], hw, ps, args.smooth)
        late_map = mean_attention_map(attns[args.late], hw, ps, args.smooth)
        rollout_map = attention_rollout(
            [attns[i] for i in rollout_blocks], hw, ps, args.smooth)
        feat_map = token_activation_map(catcher.feats['late'], hw, ps)

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
        description='Classic HiViT attention / feature overlays')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--out', default='work_dirs/vis/hivit_features.png')
    parser.add_argument(
        '--early', type=int, default=4,
        help='Early global attn block (default: 4)')
    parser.add_argument(
        '--mid', type=int, default=9,
        help='Mid global attn block (default: 9)')
    parser.add_argument(
        '--late', type=int, default=19,
        help='Late global attn / feat block (default: 19)')
    parser.add_argument(
        '--full-rollout', action='store_true',
        help='Rollout over all global_indices <= late '
             '(default: only early/mid/late)')
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
