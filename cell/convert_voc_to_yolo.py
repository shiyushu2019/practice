# -*- coding: utf-8 -*-
"""VOC -> YOLO 格式转换脚本 (doc.txt: BCCD 原始标注为 VOC 格式, 需统一转为 YOLO)。

用法:
    python convert_voc_to_yolo.py
    python convert_voc_to_yolo.py --src dataset/BCCD_Dataset --dst dataset1/BCCD_Dataset_yolo
    python convert_voc_to_yolo.py --classes RBC,WBC,Platelets   # 需要三类时显式指定

说明:
    - 输入目录需包含 Annotations/*.xml 与 JPEGImages/*.jpg (BCCD 原始结构)。
    - 输出为 {dst}/images 与 {dst}/labels, 标签为归一化中心点坐标 + 宽高 (YOLO 格式)。
    - --classes 决定保留的类别及其类序号; 不在列表中的类别 (如 Platelets) 直接丢弃。
"""
import argparse
import os
import shutil
import xml.etree.ElementTree as ET

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_voc(xml_path, class_map):
    """解析 VOC xml, 返回 [(cls_id, cx, cy, w, h), ...], 坐标已归一化到 [0, 1]。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w_img = float(size.find("width").text)
    h_img = float(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in class_map:
            continue
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        cx = (xmin + xmax) / 2 / w_img
        cy = (ymin + ymax) / 2 / h_img
        bw = (xmax - xmin) / w_img
        bh = (ymax - ymin) / h_img
        boxes.append((class_map[name], cx, cy, bw, bh))
    return boxes


def main():
    ap = argparse.ArgumentParser(description="VOC -> YOLO 格式转换")
    ap.add_argument("--src", default="dataset/BCCD_Dataset",
                    help="VOC 数据集目录 (含 Annotations 与 JPEGImages)")
    ap.add_argument("--dst", default="dataset1/BCCD_Dataset_yolo", help="输出目录")
    ap.add_argument("--classes", default="RBC,WBC",
                    help="保留的类别名 (逗号分隔, 顺序即类序号; 其余类别丢弃)")
    ap.add_argument("--ann-dir", default=None, help="标注目录 (默认 {src}/Annotations)")
    ap.add_argument("--img-dir", default=None, help="图片目录 (默认 {src}/JPEGImages)")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    class_map = {name: i for i, name in enumerate(classes)}

    ann_dir = args.ann_dir or os.path.join(args.src, "Annotations")
    img_dir = args.img_dir or os.path.join(args.src, "JPEGImages")
    dst_img = os.path.join(args.dst, "images")
    dst_lbl = os.path.join(args.dst, "labels")
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    n_img = n_box = n_skip = 0
    for xml_name in sorted(os.listdir(ann_dir)):
        if not xml_name.lower().endswith(".xml"):
            continue
        stem = os.path.splitext(xml_name)[0]
        boxes = parse_voc(os.path.join(ann_dir, xml_name), class_map)

        img_src = None
        for ext in IMG_EXTS:
            p = os.path.join(img_dir, stem + ext)
            if os.path.isfile(p):
                img_src = p
                break
        if img_src is None:
            n_skip += 1
            print(f"⚠️  跳过 {stem}: 找不到图片")
            continue

        shutil.copy2(img_src, os.path.join(dst_img, os.path.basename(img_src)))
        with open(os.path.join(dst_lbl, stem + ".txt"), "w", encoding="utf-8") as f:
            for cls, cx, cy, w, h in boxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        n_img += 1
        n_box += len(boxes)

    print(f"✅ 转换完成: {n_img} 张图片, {n_box} 个标注框 -> {args.dst}")
    if n_skip:
        print(f"   ⚠️  {n_skip} 个标注找不到对应图片, 已跳过")
    print(f"   类别映射: {class_map}")


if __name__ == "__main__":
    main()
