classes = [
    'fabric',
    'glass',
    'leather',
    'metal',
    'plastic',
    'stone',
    'wood',
]
crop_size = (
    640,
    640,
)
data_root = './data/MINC/'
dataset_type = 'BaseSegDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=3000, max_keep_ckpts=3,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=7,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=50,
        dilations=(
            1,
            1,
            2,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        style='pytorch',
        type='ResNetV1c'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        c1_channels=48,
        c1_in_channels=256,
        channels=512,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=2048,
        in_index=3,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=7,
        type='DepthwiseSeparableASPPHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
palette = [
    [
        80,
        50,
        50,
    ],
    [
        140,
        140,
        140,
    ],
    [
        204,
        5,
        255,
    ],
    [
        4,
        250,
        7,
    ],
    [
        8,
        255,
        51,
    ],
    [
        255,
        51,
        7,
    ],
    [
        255,
        6,
        51,
    ],
]
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=30000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(deterministic=False, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='./data/MINC/',
        metainfo=dict(
            classes=[
                'fabric',
                'glass',
                'leather',
                'metal',
                'plastic',
                'stone',
                'wood',
            ],
            palette=[
                [
                    80,
                    50,
                    50,
                ],
                [
                    140,
                    140,
                    140,
                ],
                [
                    204,
                    5,
                    255,
                ],
                [
                    4,
                    250,
                    7,
                ],
                [
                    8,
                    255,
                    51,
                ],
                [
                    255,
                    51,
                    7,
                ],
                [
                    255,
                    6,
                    51,
                ],
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=True,
        type='BaseSegDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=30000, type='IterBasedTrainLoop', val_interval=3000)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root='./data/MINC/',
        metainfo=dict(
            classes=[
                'fabric',
                'glass',
                'leather',
                'metal',
                'plastic',
                'stone',
                'wood',
            ],
            palette=[
                [
                    80,
                    50,
                    50,
                ],
                [
                    140,
                    140,
                    140,
                ],
                [
                    204,
                    5,
                    255,
                ],
                [
                    4,
                    250,
                    7,
                ],
                [
                    8,
                    255,
                    51,
                ],
                [
                    255,
                    51,
                    7,
                ],
                [
                    255,
                    6,
                    51,
                ],
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.4,
                    2.5,
                ),
                scale=(
                    2048,
                    1024,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.85, crop_size=(
                    640,
                    640,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(degree=20, prob=0.5, type='RandomRotate'),
            dict(
                brightness_delta=32,
                contrast_range=(
                    0.8,
                    1.2,
                ),
                hue_delta=15,
                saturation_range=(
                    0.8,
                    1.2,
                ),
                type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=True,
        type='BaseSegDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.4,
            2.5,
        ),
        scale=(
            2048,
            1024,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.85, crop_size=(
        640,
        640,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(degree=20, prob=0.5, type='RandomRotate'),
    dict(
        brightness_delta=32,
        contrast_range=(
            0.8,
            1.2,
        ),
        hue_delta=15,
        saturation_range=(
            0.8,
            1.2,
        ),
        type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='./data/MINC/',
        metainfo=dict(
            classes=[
                'fabric',
                'glass',
                'leather',
                'metal',
                'plastic',
                'stone',
                'wood',
            ],
            palette=[
                [
                    80,
                    50,
                    50,
                ],
                [
                    140,
                    140,
                    140,
                ],
                [
                    204,
                    5,
                    255,
                ],
                [
                    4,
                    250,
                    7,
                ],
                [
                    8,
                    255,
                    51,
                ],
                [
                    255,
                    51,
                    7,
                ],
                [
                    255,
                    6,
                    51,
                ],
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=True,
        type='BaseSegDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/7MINC'
