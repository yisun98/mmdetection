# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Description：用于将标注好的labelme数据集转为coco用于mmdetection 中mask_rcnn训练
"""
import os
import random
import shutil
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import pycocotools.mask
import imgviz
import numpy as np
import labelme

def write_txt(label_list):
    ori_list = ["__ignore__","_background_"]
    with open('label_power.txt', 'w') as file:
        for label in (ori_list + label_list):
            file.write('%s\n' % label)


def convert_coco(input_dir,output_dir):
    labels = "label_power.txt"
    noviz = False
    key = osp.basename(input_dir)
    if osp.exists(output_dir):
        print("Output directory already exists:", output_dir)
        sys.exit(1)
    os.makedirs(output_dir)
    os.makedirs(osp.join(output_dir, "JPEGImages"))
    if not noviz:
        os.makedirs(osp.join(output_dir, "Visualization"))
    print("Creating dataset:", output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name, )
        )
    anno_dir = osp.join(osp.dirname(output_dir), "annotations")
    os.makedirs(anno_dir,exist_ok=True)
    out_ann_file = osp.join(anno_dir , "instances_%s.json"%(key))
    label_files = glob.glob(osp.join(input_dir, "*.json"))
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(output_dir, "JPEGImages", base + ".jpg")
        out_img_file_mm = osp.join(output_dir, base + ".jpg")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        imgviz.io.imsave(out_img_file_mm, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=base + ".jpg",
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            if shape_type == "circle":
                (x1, y1), (x2, y2) = points
                r = np.linalg.norm([x2 - x1, y2 - y1])
                # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                # x: tolerance of the gap between the arc and the line segment
                n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                i = np.arange(n_points_circle)
                x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                points = np.stack((x, y), axis=1).flatten().tolist()
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not noviz:
            viz = img
            if masks:

                listdata_labels = []
                listdata_captions = []
                listdata_masks = []

                for (cnm, gid), msk in masks.items():
                    if cnm in class_name_to_id:
                        listdata_labels.append(class_name_to_id[cnm])
                        listdata_captions.append(cnm)
                        listdata_masks.append(msk)

                listdata = zip(listdata_labels, listdata_captions, listdata_masks)
                labels, captions, masks = zip(*listdata)
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    masks=masks,
                    captions=captions,
                    font_size=15,
                    line_width=2,
                )
            out_viz_file = osp.join(
                output_dir, "Visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)




def split_file(path, save_path, train_num, test_num, val_num):
    json_list = []
    label_list = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.json'):
                json_path = os.path.join(root, filename)
                json_list.append(json_path)
                with open(json_path, 'r') as file:
                    data = json.load(file)
                for data_shape in data["shapes"]:
                    img_label = data_shape["label"]
                    label_list.append(img_label)
    label_list = list(set(label_list))



    print(label_list)
    write_txt(label_list)
    random.shuffle(json_list)
    train_sum = int(len(json_list) * train_num)
    val_sum = int(len(json_list) * (train_num + val_num))
    test_sum = int(len(json_list) * (train_num + val_num + test_num))
    train_jsons = json_list[:train_sum]
    val_jsons = json_list[train_sum:val_sum]
    test_jsons = json_list[val_sum: test_sum]
    sets = {'train2017': train_jsons, 'val2017': val_jsons, 'test2017': test_jsons}
    for key, value in sets.items():
        file_save_path = os.path.join(save_path, key)
        os.makedirs(file_save_path, exist_ok=True)
        for json_path in value:
            img_path = json_path.replace(".json",".jpg")
            shutil.copy(img_path,  file_save_path)
            shutil.copy(json_path, file_save_path)

        convert_coco(file_save_path, os.path.join(save_path, "coco", key))

if __name__ == "__main__":
    path = r"data/power_raw"
    save_path = r"data/power_unoff/"
    train_num = 0.8
    test_num = 0.1
    val_num = 0.1
    split_file(path, save_path, train_num, test_num, val_num)
    print("covert finish")

