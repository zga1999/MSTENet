"""Extract backbone weights from a full detector checkpoint.

Output format matches official mmrotate LSKNet backbone checkpoints, e.g.
``lsk_s_backbone-e9d2e551.pth`` / ``lsk_t_backbone-2ef8a593.pth``::

    {'state_dict': {'patch_embed1.proj.weight': ..., 'head.weight': ..., ...}}

Keys are stripped of the ``backbone.`` prefix so they load directly via::

    init_cfg=dict(type='Pretrained', checkpoint='weights/lsk/lsk_s_dior_backbone.pth')

For LSKNet, pass ``--reference`` with an official ImageNet backbone ckpt to also
copy ``head.*`` keys (detector ckpts do not contain them).

Usage::

    python tools/extract_backbone_weights.py \\
        work_dirs/DIOR/dior_lsk-s_fpn_oriented_fix/best_dota_mAP_epoch_11.pth \\
        weights/lsk/lsk_s_dior_backbone.pth \\
        --reference weights/lsk/lsk-s_fpn_oriented_82.72.pth

    python tools/extract_backbone_weights.py \\
        work_dirs/DIOR/dior_mcst-s_fpn_oriented_fix/best_dota_mAP_epoch_12.pth \\
        weights/mcstnet/mcst_s_dior_backbone.pth
"""

import argparse
from pathlib import Path

import torch


def _load_state_dict(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return ckpt.get('state_dict', ckpt)


def extract_backbone(
    checkpoint_path: str,
    output_path: str,
    prefix: str = 'backbone.',
    reference_path: str = None,
) -> None:
    state_dict = _load_state_dict(checkpoint_path)

    backbone_sd = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        backbone_sd[key[len(prefix):]] = value

    if not backbone_sd:
        raise RuntimeError(
            f'No keys with prefix "{prefix}" found in {checkpoint_path}')

    if reference_path:
        ref_sd = _load_state_dict(reference_path)
        for key in ('head.weight', 'head.bias'):
            if key in ref_sd and key not in backbone_sd:
                backbone_sd[key] = ref_sd[key]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': backbone_sd}, out)
    print(f'extracted {len(backbone_sd)} tensors -> {out}')
    print(f'  from: {checkpoint_path}')
    if reference_path:
        print(f'  reference (head.*): {reference_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Extract backbone from detector ckpt (official LSK format)')
    parser.add_argument('checkpoint', help='Full detector checkpoint (.pth)')
    parser.add_argument('output', help='Output backbone-only checkpoint (.pth)')
    parser.add_argument(
        '--prefix', default='backbone.', help='Key prefix to strip')
    parser.add_argument(
        '--reference',
        default=None,
        help='Optional reference backbone ckpt to copy head.* keys from')
    args = parser.parse_args()
    extract_backbone(
        args.checkpoint, args.output, args.prefix, args.reference)


if __name__ == '__main__':
    main()
