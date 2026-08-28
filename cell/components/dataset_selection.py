# -*- coding: utf-8 -*-
"""数据集选择与准备工具 (BCCD / TXL-PBC / 混合)。

支持三种数据集配置:
    bccd  - 仅使用 BCCD_Dataset_preprocessed (统一类序: RBC/WBC/Platelets)
    txl   - 仅使用 TXL-PBC_Dataset (类序重映射后与 BCCD 对齐)
    mixed - 混合 BCCD_Dataset_preprocessed + TXL-PBC_preprocessed (默认)

类序统一说明:
    BCCD 标签类序:      0=RBC, 1=WBC, 2=Platelets
    TXL-PBC 标签类序:   0=WBC, 1=RBC, 2=Platelet
    为保证两个数据集可以混合使用, 统一为: 0=RBC, 1=WBC, 2=Platelets
    因此 TXL-PBC 标签需要做 0<->1 互换 (2 保持不变)。

用法:
    python dataset_selection.py                # 构建 txl 与 mixed 两套数据集
    python dataset_selection.py --all          # 同上 (显式构建全部)
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from preprocess import preprocess_image  # noqa: E402

DATASET_DIR = os.path.join(HERE, "dataset")

BCCD_SRC = os.path.join(DATASET_DIR, "BCCD_Dataset_preprocessed")
TXL_SRC = os.path.join(DATASET_DIR, "TXL-PBC_Dataset", "TXL-PBC")
TXL_DST = os.path.join(DATASET_DIR, "TXL-PBC_preprocessed")
MIXED_DST = os.path.join(DATASET_DIR, "mixed_blood_cell")

NAMES = ["RBC", "WBC", "Platelets"]  # 统一类名顺序 (0=RBC, 1=WBC, 2=Platelets)
SPLITS = ("train", "val", "test")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# TXL-PBC 原始类序 -> 统一类序: 0(WBC)->1, 1(RBC)->0, 2(Platelet)->2
TXL_REMAP = {0: 1, 1: 0, 2: 2}

DATASETS = {
    "bccd": {
        "yaml": os.path.join(DATASET_DIR, "bccd_preprocessed.yaml"),
        "test_images": os.path.join(BCCD_SRC, "test", "images"),
        "desc": "仅 BCCD_Dataset_preprocessed",
    },
    "txl": {
        "yaml": os.path.join(DATASET_DIR, "txl_pbc_preprocessed.yaml"),
        "test_images": os.path.join(TXL_DST, "test", "images"),
        "desc": "仅 TXL-PBC_Dataset",
    },
    "mixed": {
        "yaml": os.path.join(DATASET_DIR, "mixed_blood_cell.yaml"),
        "test_images": os.path.join(MIXED_DST, "test", "images"),
        "desc": "混合 BCCD + TXL-PBC",
    },
}


def _write_yaml(yaml_path, dataset_root):
    """写一份与 bccd_preprocessed.yaml 同构的数据集 yaml (path 用绝对路径, 更稳健)。"""
    content = (
        f"path: {dataset_root}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "\n"
        "nc: 3\n"
        f"names: {NAMES!r}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [yaml] {yaml_path}")


def _remap_label(src, dst, mapping):
    """把 TXL-PBC 标签文件重映射类序号后写到 dst。"""
    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        new_cls = mapping.get(cls, cls)
        out.append(" ".join([str(new_cls)] + parts[1:]))
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + ("\n" if out else ""))


def _link_or_copy(src, dst):
    """优先创建软链接, 失败则复制文件。"""
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def _unique_name(base_dir, name, source_tag):
    """处理可能的重名 (BCCD/TXL 文件名不冲突, 但保留防御逻辑)。"""
    if not os.path.exists(os.path.join(base_dir, name)):
        return name
    stem, ext = os.path.splitext(name)
    return f"{stem}_{source_tag}{ext}"


def build_txl_preprocessed(force=False):
    """构建 TXL-PBC_preprocessed: 预处理图像 (去噪+CLAHE+640×640) + 重映射标签。

    结构与 BCCD_Dataset_preprocessed 完全一致: {split}/images, {split}/labels。
    """
    print("=" * 70)
    print("构建 TXL-PBC_preprocessed (预处理图像 + 类序重映射)")
    print("=" * 70)
    if force and os.path.exists(TXL_DST):
        shutil.rmtree(TXL_DST, ignore_errors=True)

    total = 0
    for split in SPLITS:
        src_img = os.path.join(TXL_SRC, "images", split)
        src_lbl = os.path.join(TXL_SRC, "labels", split)
        if not os.path.isdir(src_img):
            print(f"  ⚠️  跳过 {split}: 找不到 {src_img}")
            continue

        dst_img = os.path.join(TXL_DST, split, "images")
        dst_lbl = os.path.join(TXL_DST, split, "labels")
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)

        images = sorted(f for f in os.listdir(src_img) if f.lower().endswith(IMG_EXTS))
        done = 0
        for name in images:
            out_img = os.path.join(dst_img, name)
            out_lbl = os.path.join(dst_lbl, os.path.splitext(name)[0] + ".txt")
            if os.path.exists(out_img) and os.path.exists(out_lbl):
                continue

            import cv2
            img = cv2.imread(os.path.join(src_img, name))
            if img is None:
                print(f"  ⚠️  无法读取 {name}, 跳过")
                continue
            cv2.imwrite(out_img, preprocess_image(img, size=640))

            src_label = os.path.join(src_lbl, os.path.splitext(name)[0] + ".txt")
            if os.path.exists(src_label):
                _remap_label(src_label, out_lbl, TXL_REMAP)
            done += 1
        total += done
        print(f"  ✅ {split}: {done} 张图片预处理完成 -> {dst_img}")

    yaml_path = DATASETS["txl"]["yaml"]
    _write_yaml(yaml_path, TXL_DST)
    print(f"✅ TXL-PBC_preprocessed 构建完成, 共处理 {total} 张图片\n")
    return TXL_DST


def build_mixed(force=False):
    """构建 mixed_blood_cell: 合并 BCCD_Dataset_preprocessed + TXL-PBC_preprocessed。

    两个数据源均已统一为 640×640、类序 RBC/WBC/Platelets, 因此直接软链接合并。
    """
    print("=" * 70)
    print("构建 mixed_blood_cell (BCCD + TXL-PBC 混合)")
    print("=" * 70)
    if not os.path.exists(TXL_DST):
        print("  ⚠️  未找到 TXL-PBC_preprocessed, 先构建它 ...")
        build_txl_preprocessed(force=force)

    if force and os.path.exists(MIXED_DST):
        shutil.rmtree(MIXED_DST, ignore_errors=True)

    sources = (("bccd", BCCD_SRC), ("txl", TXL_DST))
    for split in SPLITS:
        dst_img = os.path.join(MIXED_DST, split, "images")
        dst_lbl = os.path.join(MIXED_DST, split, "labels")
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)

        n_img = n_lbl = 0
        for tag, src_root in sources:
            src_img = os.path.join(src_root, split, "images")
            src_lbl = os.path.join(src_root, split, "labels")
            if not os.path.isdir(src_img):
                continue
            for name in sorted(os.listdir(src_img)):
                if not name.lower().endswith(IMG_EXTS):
                    continue
                dst_name = _unique_name(dst_img, name, tag)
                _link_or_copy(os.path.join(src_img, name), os.path.join(dst_img, dst_name))
                n_img += 1

                label_name = os.path.splitext(name)[0] + ".txt"
                src_label = os.path.join(src_lbl, label_name)
                if os.path.exists(src_label):
                    dst_label = os.path.splitext(dst_name)[0] + ".txt"
                    _link_or_copy(src_label, os.path.join(dst_lbl, dst_label))
                    n_lbl += 1
        print(f"  ✅ {split}: 合并 {n_img} 张图片 / {n_lbl} 个标签 -> {dst_img}")

    _write_yaml(DATASETS["mixed"]["yaml"], MIXED_DST)
    print("✅ mixed_blood_cell 构建完成\n")
    return MIXED_DST


def resolve_yaml(name):
    """返回数据集名称对应的 yaml 路径。"""
    if name not in DATASETS:
        raise ValueError(f"未知数据集: {name}, 可选 {sorted(DATASETS)}")
    return DATASETS[name]["yaml"]


def ensure_dataset_ready(name):
    """确保目标数据集已构建 (bccd 直接使用现有预处理目录)。"""
    if name == "bccd":
        yaml_path = DATASETS["bccd"]["yaml"]
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"缺少 BCCD 数据集 yaml: {yaml_path}")
        return yaml_path

    if name == "txl":
        yaml_path = DATASETS["txl"]["yaml"]
        if not (os.path.exists(TXL_DST) and os.path.exists(yaml_path)):
            build_txl_preprocessed()
        return yaml_path

    if name == "mixed":
        yaml_path = DATASETS["mixed"]["yaml"]
        if not (os.path.exists(MIXED_DST) and os.path.exists(yaml_path)):
            build_mixed()
        return yaml_path

    raise ValueError(f"未知数据集: {name}, 可选 {sorted(DATASETS)}")


def resolve_test_images(name):
    """返回数据集的测试集图片目录 (供推理默认使用)。"""
    if name not in DATASETS:
        raise ValueError(f"未知数据集: {name}, 可选 {sorted(DATASETS)}")
    return DATASETS[name]["test_images"]


def main():
    ap = argparse.ArgumentParser(description="构建 TXL-PBC_preprocessed 与 mixed_blood_cell 数据集")
    ap.add_argument("--all", action="store_true", help="构建全部 (txl + mixed)")
    ap.add_argument("--force", action="store_true", help="删除已存在目录后重建")
    ap.add_argument("--dataset", default=None, choices=["txl", "mixed"],
                    help="仅构建指定数据集 (默认构建 txl 和 mixed)")
    args = ap.parse_args()

    if args.dataset == "txl":
        build_txl_preprocessed(force=args.force)
    elif args.dataset == "mixed":
        build_mixed(force=args.force)
    else:
        build_txl_preprocessed(force=args.force)
        build_mixed(force=args.force)


if __name__ == "__main__":
    main()
