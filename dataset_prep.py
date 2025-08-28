
# dataset_prep.py

import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

random.seed(42)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def convert_bbox_voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h

def save_yolo_label_file(label_path: str, objects: List[Tuple[int, float, float, float, float]]):
    with open(label_path, "w") as f:
        for obj in objects:
            class_id, x, y, w, h = obj
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def create_yolo_dataset_from_voc(voc_annotations_dir: str, images_dir: str, output_dir: str, class_map: dict, val_fraction=0.15):
    import xml.etree.ElementTree as ET
    ensure_dir(output_dir)
    yolo_images = os.path.join(output_dir, "images")
    yolo_labels = os.path.join(output_dir, "labels")
    ensure_dir(yolo_images)
    ensure_dir(yolo_labels)

    xml_files = [f for f in os.listdir(voc_annotations_dir) if f.endswith(".xml")]
    dataset = []
    for xml in xml_files:
        tree = ET.parse(os.path.join(voc_annotations_dir, xml))
        root = tree.getroot()
        filename = root.find("filename").text
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        objects = []
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in class_map:
                continue
            class_id = class_map[cls]
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            x, y, bw, bh = convert_bbox_voc_to_yolo(xmin, ymin, xmax, ymax, w, h)
            objects.append((class_id, x, y, bw, bh))

        if len(objects) == 0:
            continue

        src_img = os.path.join(images_dir, filename)
        if not os.path.exists(src_img):
            continue
        dst_img = os.path.join(yolo_images, filename)
        shutil.copy2(src_img, dst_img)

        label_name = os.path.splitext(filename)[0] + ".txt"
        save_yolo_label_file(os.path.join(yolo_labels, label_name), objects)
        dataset.append(filename)

    random.shuffle(dataset)
    val_count = int(len(dataset) * val_fraction)
    val_set = set(dataset[:val_count])
    with open(os.path.join(output_dir, "train.txt"), "w") as ftrain,          open(os.path.join(output_dir, "val.txt"), "w") as fval:
        for fn in dataset:
            path = os.path.abspath(os.path.join(yolo_images, fn))
            if fn in val_set:
                fval.write(path + "\n")
            else:
                ftrain.write(path + "\n")

    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, "w") as f:
        for cls in sorted(class_map, key=lambda k: class_map[k]):
            f.write(cls + "\n")

    print("YOLO dataset created at", output_dir)
