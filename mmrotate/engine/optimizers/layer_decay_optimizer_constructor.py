# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Layer-wise learning rate decay for ViT / HiViT backbones.

Ported from the mmrotate 0.x ``mmcv_custom`` package
(``layer_decay_optimizer_constructor.py`` and
``layer_decay_optimizer_constructor_hivit.py``), which is itself mostly
copy-paste from BEiT:
https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/mmcv_custom/layer_decay_optimizer_constructor.py

The 1.x adaptation only swaps ``mmcv.runner`` for its mmengine counterparts
(``DefaultOptimWrapperConstructor``, ``get_dist_info``,
``OPTIM_WRAPPER_CONSTRUCTORS``) and routes the printing through
:class:`~mmengine.logging.MMLogger`. The grouping rules are unchanged, and the
ViT and HiViT variants deliberately keep their own (slightly different)
no-decay predicates.
"""
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor

from mmrotate.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_num_layer_for_vit(var_name, num_max_layer):
    """Map a parameter name to its ViT layer id."""
    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.blocks'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1


def get_hivit_layer_id(name, num_layers, main_block):
    """Map a parameter name to its HiViT layer id.

    HiViT interleaves blocks without attention, so the layer id is taken from
    the position of the nearest preceding "main" block.
    """
    if name.startswith('backbone.'):
        name = name[9:]
        if name in ['cls_token', 'pos_embed', 'absolute_pos_embed']:
            return 0
        elif name.startswith('patch_embed'):
            return 0
        elif name.startswith('blocks'):
            i = int(name.split('.')[1])
            while i >= 0:
                try:
                    return main_block.index(i) + 1
                except ValueError:
                    i = i - 1
            return 0
    return num_layers - 1


def _parse_num_layers(paramwise_cfg):
    """Read ``(num_layers, layer_decay_rate)`` out of ``paramwise_cfg``."""
    if isinstance(paramwise_cfg, dict):
        num_layers = paramwise_cfg.get('num_layers')
        layer_decay_rate = paramwise_cfg.get('layer_decay_rate')
    else:
        assert len(paramwise_cfg) == 2
        num_layers, layer_decay_rate = paramwise_cfg
    return num_layers + 2, layer_decay_rate


def _build_param_groups(module, num_layers, layer_decay_rate, base_wd, base_lr,
                        is_no_decay, layer_id_fn, logger):
    """Group parameters by layer id and weight-decay eligibility."""
    parameter_groups = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if is_no_decay(name, param):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = base_wd

        layer_id = layer_id_fn(name, num_layers)
        group_name = f'layer_{layer_id}_{group_name}'

        if group_name not in parameter_groups:
            scale = layer_decay_rate**(num_layers - layer_id - 1)

            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': scale,
                'group_name': group_name,
                'lr': scale * base_lr,
            }

        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

    rank, _ = get_dist_info()
    if rank == 0:
        logger.info(
            f'LayerDecay: {len(parameter_groups)} param groups '
            f'(num_layers={num_layers}, decay={layer_decay_rate})')

    return parameter_groups


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Layer-wise lr decay for a plain ViT backbone."""

    def _validate_cfg(self):
        if 'custom_keys' in self.paramwise_cfg:
            if not isinstance(self.paramwise_cfg['custom_keys'], dict):
                raise TypeError(
                    'If specified, custom_keys must be a dict, '
                    f'but got {type(self.paramwise_cfg["custom_keys"])}')
            if self.base_wd is None:
                for key in self.paramwise_cfg['custom_keys']:
                    if 'decay_mult' in self.paramwise_cfg['custom_keys'][key]:
                        raise ValueError('base_wd should not be None')

        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in self.paramwise_cfg
                or 'norm_decay_mult' in self.paramwise_cfg
                or 'dwconv_decay_mult' in self.paramwise_cfg):
            if self.base_wd is None:
                raise ValueError('base_wd should not be None')

    @staticmethod
    def _is_no_decay(name, param):
        return (len(param.shape) == 1 or name.endswith('.bias')
                or name.endswith('_token') or name.endswith('pos_embed'))

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of ``module`` to the param group list."""
        logger = MMLogger.get_current_instance()
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers, layer_decay_rate = _parse_num_layers(self.paramwise_cfg)
        logger.info('Build LayerDecayOptimizerConstructor '
                    f'{layer_decay_rate:f} - {num_layers:d}')

        parameter_groups = _build_param_groups(
            module,
            num_layers,
            layer_decay_rate,
            self.base_wd,
            self.base_lr,
            is_no_decay=self._is_no_decay,
            layer_id_fn=get_num_layer_for_vit,
            logger=logger)
        params.extend(parameter_groups.values())


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class HiViTLayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Layer-wise lr decay for HiViT; main blocks are those with attention."""

    #: attribute a block must expose to count as a "main" block
    main_block_attr = 'attn'

    def _validate_cfg(self):
        pass

    @staticmethod
    def _is_no_decay(name, param):
        return (len(param.shape) == 1 or name.endswith('.bias')
                or name in ('pos_embed', 'cls_token', 'absolute_pos_embed')
                or 'relative_position_bias_table' in name)

    def add_params(self, params, module, **kwargs):
        """Add all parameters of ``module`` to the param group list."""
        logger = MMLogger.get_current_instance()
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        logger.info(f'Build {self.__class__.__name__}')
        num_layers, layer_decay_rate = _parse_num_layers(self.paramwise_cfg)

        main_block = [
            i for i, blk in enumerate(module.backbone.blocks)
            if getattr(blk, self.main_block_attr, None) is not None
        ]

        parameter_groups = _build_param_groups(
            module,
            num_layers,
            layer_decay_rate,
            self.base_wd,
            self.base_lr,
            is_no_decay=self._is_no_decay,
            layer_id_fn=lambda name, n: get_hivit_layer_id(
                name, n, main_block),
            logger=logger)
        params.extend(parameter_groups.values())


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class HiViT2LayerDecayOptimizerConstructor(HiViTLayerDecayOptimizerConstructor
                                           ):
    """Same as above, but main blocks are those carrying an mlp."""

    main_block_attr = 'mlp'
