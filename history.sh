1. 安装文档
https://mmdetection.readthedocs.io/zh-cn/latest/get_started.html
mim install mmcv==2.1.0
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
python demo/image_demo.py demo/demo.jpg configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --out-file demo_detection.jpg
CUDA_VISIBLE_DEVICES=1  python tools/train.py configs/balloon/mask_rcnn_r50_fpn_2x_coco_balloon.py
python tools/test.py configs/balloon/mask_rcnn_r50_fpn_2x_coco_balloon.py  work_dir/mask_rcnn_r50_fpn_2x_coco_balloon/epoch_24.pth 


python demo/image_demo.py data/ballon/val/410488422_5f8991f26e_b.jpg configs/balloon/mask_rcnn_r50_fpn_2x_coco_balloon.py work_dir/mask_rcnn_r50_fpn_2x_coco_balloon/epoch_24.pth --out-file ballon_410488422_5f8991f26e_b.jpg


pip install imgviz  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install labelme  -i https://pypi.tuna.tsinghua.edu.cn/simple

https://github.com/labelmeai/labelme

CUDA_VISIBLE_DEVICES=1  python tools/train.py configs/power/mask_rcnn_r50_fpn_2x_coco_power.py