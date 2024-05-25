# 新配置继承了基本配置，并做了必要的修改
# _base_ = '../mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py'
_base_ = ['../mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py']

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5),
        mask_head=dict(num_classes=5)),
    backbone=dict(init_cfg=None),
)

# 修改数据集相关配置
data_root = 'data/power_off_coco/'
metainfo = {
    'classes': ('并沟线夹', '耐张线夹', '玻璃绝缘子', '复合绝缘子', '导线悬垂线夹发热', ),
    'palette': [
        (220, 20, 60),
        (119, 11, 32), 
        (0, 0, 230),
          (106, 0, 228),
         (0, 60, 100),
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/JPEGImages/annotations.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/JPEGImages/annotations.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'val/JPEGImages/annotations.json')
test_evaluator = val_evaluator

runner = dict(max_epochs=24, type='EpochBasedRunner')

work_dir = 'work_dir/mask_rcnn_r50_fpn_2x_coco_power'

# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
load_from = 'checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'