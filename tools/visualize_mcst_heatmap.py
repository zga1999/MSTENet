"""
python tools/visualize_mcst_heatmap.py \
    --config projects/mcst/dior_mcst-s_fpn_oriented_1x.py \
    --checkpoint work_dirs/DIOR/dior_mcst-s_fpn_oriented/best_dota_mAP_epoch_12.pth \
    --img-path demo/P0382__1024__59___148.png \
    --out-dir work_vis/heap_map
"""

import argparse
import os
import os.path as osp
from typing import Sequence

import cv2
import mmcv
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

import mmrotate  # noqa: F401
from mmrotate.registry import VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature heatmaps')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img-path', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--channel-reduction', default='select_max') #choices=['select_max', 'mean', 'topk', 'None']
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--arrangement', nargs=2, type=int, default=[2, 2], metavar=('ROWS', 'COLS'))
    parser.add_argument('--target-layers', nargs='+', default=['backbone'])
    parser.add_argument('--preview-model', action='store_true')
    return parser.parse_args()


def get_file_list(path):
    """Get image file list from one image path or a directory."""
    source_type = dict(is_dir=False)
    if osp.isdir(path):
        source_type['is_dir'] = True
        image_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_list = []
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(image_ext):
                image_list.append(osp.join(path, name))
        return image_list, source_type
    if osp.isfile(path):
        return [path], source_type
    raise FileNotFoundError(f'IMG_PATH does not exist: {path}')


def auto_arrange_images(images, arrangement):
    """Arrange images to a fixed-size grid."""
    if not images:
        return None
    rows, cols = arrangement
    first = images[0]
    h, w = first.shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        if idx < len(images):
            img = images[idx]
            if img.shape[:2] != (h, w):
                img = mmcv.imresize(img, (w, h))
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return canvas


def _to_numpy_featmap(featmap):
    if isinstance(featmap, torch.Tensor):
        return featmap.detach().cpu().float().numpy()
    if isinstance(featmap, np.ndarray):
        return featmap.astype(np.float32)
    return None


def draw_featmap(featmap, base_img, channel_reduction='select_max', topk=4):
    """Draw a heatmap from one feature map tensor and overlay on image."""
    feat = _to_numpy_featmap(featmap)
    if feat is None:
        return base_img

    if feat.ndim == 4:
        feat = feat[0]
    if feat.ndim != 3:
        return base_img

    if channel_reduction is None or channel_reduction == 'mean':
        fm = feat.mean(axis=0)
    elif channel_reduction == 'select_max':
        fm = feat.max(axis=0)
    elif channel_reduction == 'topk':
        k = max(1, min(int(topk), feat.shape[0]))
        score = feat.reshape(feat.shape[0], -1).mean(axis=1)
        idx = np.argsort(score)[-k:]
        fm = feat[idx].mean(axis=0)
    else:
        fm = feat.max(axis=0)

    fm -= fm.min()
    denom = fm.max()
    if denom > 1e-6:
        fm /= denom
    fm = (fm * 255).astype(np.uint8)

    heatmap = mmcv.imresize(fm, (base_img.shape[1], base_img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return (0.5 * base_img + 0.5 * heatmap).astype(np.uint8)


class ActivationsWrapper:

    def __init__(self, model, target_layers):
        self.model = model
        self.activations = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def __call__(self, img_path):
        self.activations = []
        with torch.no_grad():
            results = inference_detector(self.model, img_path)
        return results, self.activations

    def release(self):
        for handle in self.handles:
            handle.remove()


def draw_result(visualizer, image_path, result, score_thr):
    """Draw predictions like 0.x ``model.show_result``; returns BGR."""
    img = mmcv.imconvert(mmcv.imread(image_path), 'bgr', 'rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        show=False,
        pred_score_thr=score_thr)
    return mmcv.imconvert(visualizer.get_image(), 'rgb', 'bgr')


def main():
    args = parse_args()
    channel_reduction = args.channel_reduction
    if channel_reduction == 'None':
        channel_reduction = None
    assert len(args.arrangement) == 2

    init_default_scope('mmrotate')
    model = init_detector(args.config, args.checkpoint, device=args.device)

    if not args.show:
        os.makedirs(args.out_dir, exist_ok=True)

    if args.preview_model:
        print(model)
        print('\n This flag is only to show the model. If you want to continue, '
              'please remove --preview-model.')
        return

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(eval(f'model.{target_layer}'))
        except Exception as e:
            print(model)
            raise RuntimeError('Layer does not exist', e)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    activations_wrapper = ActivationsWrapper(model, target_layers)

    image_list, source_type = get_file_list(args.img_path)

    progress_bar = ProgressBar(len(image_list))
    for image_path in image_list:
        result, featmaps = activations_wrapper(image_path)
        if not isinstance(featmaps, Sequence):
            featmaps = [featmaps]

        flatten_featmaps = []
        for featmap in featmaps:
            if isinstance(featmap, Sequence):
                flatten_featmaps.extend(featmap)
            else:
                flatten_featmaps.append(featmap)

        if source_type['is_dir']:
            filename = os.path.relpath(image_path, args.img_path).replace('/', '_')
        else:
            filename = os.path.basename(image_path)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        # 先画检测结果，再叠加特征图
        drawn_img = draw_result(visualizer, image_path, result, args.score_thr)

        shown_imgs = []
        for featmap in flatten_featmaps:
            shown_img = draw_featmap(
                featmap,
                drawn_img,
                channel_reduction=channel_reduction,
                topk=args.topk)
            shown_imgs.append(shown_img)

        shown_imgs = auto_arrange_images(shown_imgs, args.arrangement)
        if shown_imgs is None:
            shown_imgs = drawn_img

        progress_bar.update()
        if out_file:
            mmcv.imwrite(shown_imgs, out_file)
            print(f'{out_file} has been saved')

        if args.show:
            mmcv.imshow(shown_imgs, win_name='featmap', wait_time=0)

    activations_wrapper.release()
    if not args.show:
        print(f'All done!'
              f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
