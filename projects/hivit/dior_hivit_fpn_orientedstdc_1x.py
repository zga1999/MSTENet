""""

python tools/train.py projects/hivit/dior_hivit_fpn_orientedstdc_1x.py \
        --work-dir work_dirs/test

python tools/test.py projects/hivit/dior_hivit_fpn_orientedstdc_1x.py \
        --work-dir work_dirs/DIOR/dior_hivit_fpn_orientedstdc


CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh projects/hivit/dior_hivit_fpn_orientedstdc_1x.py 2 \
        --work-dir work_dirs/DIOR/dior_hivit_fpn_orientedstdc

CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh projects/hivit/dior_hivit_fpn_orientedstdc_1x.py 4 \
        --work-dir work_dirs/DIOR/dior_hivit_fpn_orientedstdc

""""

######################################## 全局配置 ########################################
default_scope = 'mmrotate'
angle_version = 'le90'
dataset_type = 'DIORDataset'
data_root = '../datasets/obb/DIOR/'
backend_args = None

####################################### 数据流水线配置 #######################################
# DIOR 原图多为 800x800，与 configs/_base_/datasets/dior.py 保持一致
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # DIOR 的 04137/07007 各有一个高为 0 的退化标注，编码时 log(0) 会让
    # loss_bbox 变成 nan 并污染全部权重，必须在进网络前剔除
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1e-2, 1e-2),
        keep_empty=True),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

####################################### 数据加载器配置 #######################################
# 目录结构:
#   ImageSets/Main/train.txt val.txt trainval.txt test.txt
#   JPEGImages-trainval / JPEGImages-test
#   Annotations/Oriented Bounding Boxes/*.xml  (DIORDataset 默认 ann_subdir)
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['DATASET_TYPE'],
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='ImageSets/Main/train.txt',
                data_prefix=dict(img_path='JPEGImages-trainval'),
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=train_pipeline,
                backend_args=backend_args),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='ImageSets/Main/val.txt',
                data_prefix=dict(img_path='JPEGImages-trainval'),
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=train_pipeline,
                backend_args=backend_args)
        ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/test.txt',
        data_prefix=dict(img_path='JPEGImages-test'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

####################################### 评估器配置 #######################################
val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

####################################### 训练/测试循环配置 #######################################
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

####################################### 优化器与学习率配置 #######################################
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='HiViTLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=20, layer_decay_rate=0.9),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[9, 11],
        gamma=0.1)
]

####################################### 运行时配置 #######################################
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=1,
        save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

####################################### 模型配置 #######################################
# MAE full ckpt split: encoder → backbone; decoder → MAE bbox head.
MAE_pretrained = 'weights/hivit/mae_bboxhead.pth'
backbone_pretrained = 'weights/hivit/hivit_backbone.pth'
model = dict(
    type='RotatedimTED',
    proposals_dim=6,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='HiViT',
        img_size=224,
        patch_size=16,
        embed_dim=512,
        depths=[2, 2, 20],
        num_heads=8,
        mlp_ratio=4.,
        rpe=False,
        drop_path_rate=0.1,
        with_fpn=True,
        out_indices=['H', 'M', 19, 19],
        use_checkpoint=False,
        global_indices=[4, 9, 14, 19],
        window_size=14,
        init_cfg=dict(type='Pretrained', checkpoint=backbone_pretrained),
        last_feat=True),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[128, 256, 512, 512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            use_box_type=True),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_version=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=1.0)),
    roi_skip_fpn=False,
    with_mfm=True,
    roi_head=dict(
        type='OrientedStandardRoIHeadimTED',
        bbox_roi_extractor=[
            dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=7,
                    sample_num=2,
                    clockwise=True),
                out_channels=512,
                featmap_strides=[4, 8, 16, 32]),
            dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=7,
                    sample_num=2,
                    clockwise=True),
                out_channels=512,
                featmap_strides=[16])
        ],
        bbox_head=dict(
            type='RotatedMAEBBoxHeadSTDC',
            predict_box_type='rbox',
            init_cfg=dict(type='Pretrained', checkpoint=MAE_pretrained),
            use_checkpoint=False,
            dc_mode_str_list=['', '', '', 'XY', '', 'A', '', 'WH'],
            num_convs_list=[0, 0, 3, 3, 2, 2, 1, 1],
            am_mode_str_list=['', '', 'V', 'V', 'V', 'V', 'V', 'V'],
            rois_mode='rbbox',
            in_channels=512,
            img_size=224,
            patch_size=16,
            embed_dim=512,
            depth=8,
            num_heads=16,
            mlp_ratio=4.,
            num_classes=20,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                gpu_assign_thr=800,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBbox2HBboxOverlaps2D')),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                gpu_assign_thr=800,
                ignore_iof_thr=-1),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)))
