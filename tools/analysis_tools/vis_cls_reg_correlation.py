"""收集「RoI 分类分数 vs 回归前/回归后 IoU」并绘制相关性密度散点图（mmdet 3.x / mmengine）。

从 mmrotate 0.x 的同名脚本移植而来。0.x 的
``rpn_head.simple_test_rpn`` → 1.x 的 ``rpn_head.predict``，
``build_dataset/build_dataloader`` → ``Runner.build_dataloader``，
gt 从 ``data['gt_bboxes']`` 改成 ``data_samples[i].gt_instances``。

用法::

    python tools/analysis_tools/vis_cls_reg_correlation.py \
        projects/vit/dior_vit_fpn_orientedstdcbbn_dualfocal_1x.py \
        work_dirs/DIOR/dior_vit_fpn_orientedstdcbbn_dualfocal/best_dota_mAP_epoch_12.pth \
        --out-dir work_vis/cls_reg_correlation \
        --tag dior_test --max-imgs 200 --score-mode matched --drop-degenerate

原始数据缓存为 raw_<tag>.npz，之后只调样式不再重跑推理::

    python tools/analysis_tools/plot_cls_iou_corr.py \
        work_vis/cls_reg_correlation/raw_dior_test.npz \
        --out-dir work_vis/cls_reg_correlation --tag dior_test --drop-degenerate
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from mmdet.structures.bbox import bbox2roi
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbbox_overlaps

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from plot_cls_iou_corr import add_style_args, plot_pair, style_kwargs


def _as_tensor(boxes):
    """RotatedBoxes / Tensor → (N, 5) Tensor。"""
    return boxes.tensor if hasattr(boxes, 'tensor') else boxes


class Collector:
    """累积三个等长数组：分类分数、回归前 IoU、回归后 IoU。"""

    def __init__(self):
        self.cls_scores, self.iou_in, self.iou_out = [], [], []

    def append(self, s, i_in, i_out):
        self.cls_scores.append(s.float().cpu().numpy())
        self.iou_in.append(i_in.float().cpu().numpy())
        self.iou_out.append(i_out.float().cpu().numpy())

    def arrays(self):
        if not self.cls_scores:
            e = np.asarray([], np.float32)
            return e, e, e
        return (np.concatenate(self.cls_scores),
                np.concatenate(self.iou_in),
                np.concatenate(self.iou_out))


@torch.no_grad()
def collect_one_batch(model, data_batch, collector,
                      score_mode='matched', iou_thr=0.0):
    """对一个 batch 做 RPN + RoI head 前向，抽出 (分数, IoU_in, IoU_out)。

    score_mode:
      'matched' —— 取「与该 RoI 重叠最大的那个 GT 所属类别」的 softmax 概率。
                   这是参考文献里 s 的常规定义，能避开 max 模式下被无关
                   类别顶满导致的 s≈1 饱和带。
      'max'     —— 取所有前景类别 softmax 概率的最大值（0.x 脚本的旧行为）。
    """
    data = model.data_preprocessor(data_batch, False)
    inputs, data_samples = data['inputs'], data['data_samples']

    # RPN 走 FPN 金字塔，RoI head 可能额外吃 ViF 特征，交给模型自己拆
    x, roi_feats = model._split_feats(inputs)
    rpn_results_list = model.rpn_head.predict(x, data_samples, rescale=False)

    for i, (data_sample, rpn_results) in enumerate(
            zip(data_samples, rpn_results_list)):
        gt_instances = data_sample.gt_instances
        gt_bboxes = _as_tensor(gt_instances.bboxes)
        if gt_bboxes.numel() == 0 or len(rpn_results) == 0:
            continue

        proposals = rpn_results.bboxes                 # RotatedBoxes (N, 5)
        proposals_t = _as_tensor(proposals)

        # ── 回归前 IoU：RPN proposal 与 GT 的重叠 ──────────────────────
        overlaps_in = rbbox_overlaps(proposals_t, gt_bboxes)     # (N, G)
        iou_in, matched_gt = overlaps_in.max(dim=1)

        # ── RoI head 前向（分支 C，即测试时真正用的那支）──────────────
        rois = bbox2roi([proposals])                   # (N, 6) 带 batch_idx
        rois[:, 0] = i
        bbox_results = model.roi_head._bbox_forward(roi_feats, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        bbox_head = getattr(model.roi_head, 'bbox_head_c',
                            model.roi_head.bbox_head)

        # ── 分类分数 ──────────────────────────────────────────────────
        prob = torch.softmax(cls_score, dim=-1)[:, :-1]          # 去掉背景列
        if score_mode == 'matched':
            target_cls = gt_instances.labels[matched_gt]
            score = prob[torch.arange(len(target_cls),
                                      device=prob.device), target_cls]
        else:
            score = prob.max(dim=1).values

        # ── 回归后 IoU：解码预测框再与 GT 求重叠 ──────────────────────
        if bbox_pred.shape[1] == 5:                    # class-agnostic
            deltas = bbox_pred
        else:                                          # class-specific
            pick = target_cls if score_mode == 'matched' else prob.argmax(1)
            deltas = bbox_pred.view(len(pick), -1, 5)[
                torch.arange(len(pick), device=pick.device), pick]
        decoded = _as_tensor(bbox_head.bbox_coder.decode(proposals_t, deltas))

        # 与「同一个」GT 比较，才能反映这个 RoI 的定位是变好还是变差
        iou_out = rbbox_overlaps(
            decoded, gt_bboxes, is_aligned=False).gather(
                1, matched_gt[:, None]).squeeze(1)

        keep = iou_in > iou_thr
        if keep.sum() == 0:
            continue
        collector.append(score[keep], iou_in[keep], iou_out[keep])


def parse_args():
    parser = argparse.ArgumentParser(
        description='统计 RoI 分类分数与回归前/后 IoU 的相关性并出图')
    parser.add_argument('config', help='模型配置文件')
    parser.add_argument('checkpoint', help='checkpoint (.pth)')
    parser.add_argument('--out-dir', default='work_vis/cls_reg_correlation')
    parser.add_argument('--tag', default='', help='输出文件名后缀')
    parser.add_argument('--max-imgs', type=int, default=200,
                        help='最多分析多少张验证图')
    parser.add_argument('--score-mode', default='matched',
                        choices=['matched', 'max'],
                        help='matched=匹配 GT 类别的概率（推荐）；max=全类别最大值')
    parser.add_argument('--iou-thr', type=float, default=0.0,
                        help='收集时的最低 IoU_in（0=全保留，过滤留到绘图阶段）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--style', default='density',
                        choices=['density', 'none'],
                        help='none=只收集数据存 npz，不出图')
    add_style_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    cfg.load_from = None          # 权重由 --checkpoint 指定，别让配置里的 load_from 干扰

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.to(args.device).eval()
    print(f'[✓] 模型已加载: {args.checkpoint}')

    # val_pipeline 里带 LoadAnnotations，所以验证集能拿到 gt_instances
    dl_cfg = cfg.val_dataloader.copy()
    dl_cfg['batch_size'] = 1
    dl_cfg['num_workers'] = 2
    loader = Runner.build_dataloader(dl_cfg)
    n_imgs = min(args.max_imgs, len(loader))
    print(f'[✓] 验证集 {len(loader)} 张，分析前 {n_imgs} 张，'
          f'score_mode={args.score_mode}')

    collector = Collector()
    for i, data_batch in enumerate(loader):
        if i >= args.max_imgs:
            break
        try:
            collect_one_batch(model, data_batch, collector,
                              score_mode=args.score_mode,
                              iou_thr=args.iou_thr)
        except Exception as e:
            print(f'  [!] 第 {i} 张跳过: {type(e).__name__}: {e}')
        if (i + 1) % 20 == 0:
            print(f'  已处理 {i + 1} / {n_imgs} ...')

    cls_scores, iou_in, iou_out = collector.arrays()
    if len(cls_scores) == 0:
        raise RuntimeError('没有收集到任何 RoI 样本，请检查数据流水线与模型前向。')
    print(f'\n[✓] 共收集 {len(cls_scores):,} 个 RoI 样本')

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f'_{args.tag}' if args.tag else ''
    npz_path = osp.join(args.out_dir, f'raw{suffix}.npz')
    np.savez_compressed(npz_path, cls_scores=cls_scores,
                        iou_in=iou_in, iou_out=iou_out)
    print(f'[✓] 原始数据已缓存: {npz_path}')

    if args.style == 'density':
        plot_pair(cls_scores, iou_in, iou_out, args.out_dir,
                  tag=args.tag, **style_kwargs(args))


if __name__ == '__main__':
    main()
