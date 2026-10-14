"""统一验证脚本 —— 在测试集/验证集上评估所有训练权重。

用法:
    python eval.py                    # 默认测试集, 自动选卡, 混合数据集
    python eval.py --split val        # 用验证集
    python eval.py --device 6         # 换卡
    python eval.py --dataset bccd     # 仅 BCCD 数据集
    python eval.py --dataset txl      # 仅 TXL-PBC 数据集
    python eval.py --dataset mixed    # 混合两个数据集 (默认)
"""
import argparse
import os
import re
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from components.dataset_selection import resolve_yaml, ensure_dataset_ready  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "weights")
RESULTS_DIR = os.path.join(HERE, "results")
TEMP_DIR = os.path.join(HERE, "temp")
DOC_PATH = os.path.join(HERE, "doc.txt")

MODELS = {
    "baseline": os.path.join(WEIGHTS_DIR, "baseline.pt"),
    "focal": os.path.join(WEIGHTS_DIR, "focal.pt"),
    "cbam": os.path.join(WEIGHTS_DIR, "cbam.pt"),
    "dams": os.path.join(WEIGHTS_DIR, "dams.pt"),
    "cf": os.path.join(WEIGHTS_DIR, "cf.pt"),
}
NAMES = ["RBC", "WBC"]


def load_document_metrics():
    """读取 doc.txt 中 CF-YOLOv8 的 Precision 和 mAP@0.5 声明。"""
    with open(DOC_PATH, encoding="utf-8") as f:
        document = f.read()
    match = re.search(
        r"改进后的\s*CF-YOLOv8.*?精确率达\s*([0-9]+(?:\.[0-9]+)?)%.*?"
        r"mAP@0\.5\s*达\s*([0-9]+(?:\.[0-9]+)?)",
        document,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"无法从 {DOC_PATH} 读取 CF-YOLOv8 的 Precision/mAP@0.5 声明")
    return {
        "precision": float(match.group(1)) / 100,
        "map50": float(match.group(2)),
    }


def save_accuracy_comparison(records, args, document_metrics):
    """保存实测指标与文档声明的可视化对比表。"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    output_path = os.path.join(
        TEMP_DIR, f"accuracy_comparison_{args.dataset}_{args.split}.png"
    )
    columns = [
        "Model",
        "Measured Precision",
        "Doc Precision",
        "Precision Delta",
        "Measured mAP@0.5",
        "Doc mAP@0.5",
        "mAP Delta",
    ]
    rows = []
    for record in records:
        rows.append([
            record["name"],
            f'{record["precision"]:.4f} ({record["precision"]:.2%})',
            f'{document_metrics["precision"]:.4f} ({document_metrics["precision"]:.2%})',
            f'{record["precision"] - document_metrics["precision"]:+.4f}',
            f'{record["map50"]:.4f}',
            f'{document_metrics["map50"]:.4f}',
            f'{record["map50"] - document_metrics["map50"]:+.4f}',
        ])

    if not rows:
        print("⚠️  没有成功评估的模型，跳过准确率对比图")
        return

    figure, axis = plt.subplots(figsize=(16, max(2.8, 0.65 * len(rows) + 1.8)))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.11, 0.17, 0.15, 0.13, 0.17, 0.14, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#B8C2CC")
    for column_index in range(len(columns)):
        table[0, column_index].set_facecolor("#2F5597")
        table[0, column_index].set_text_props(color="white", weight="bold")
    for row_index in range(1, len(rows) + 1):
        color = "#F3F6FA" if row_index % 2 else "white"
        for column_index in range(len(columns)):
            table[row_index, column_index].set_facecolor(color)

    axis.set_title(
        f"Model Accuracy Comparison ({args.dataset}, {args.split})\n"
        f"Document source: doc.txt | CF-YOLOv8 claim: "
        f"Precision {document_metrics['precision']:.2%}, "
        f"mAP@0.5 {document_metrics['map50']:.3f}",
        fontsize=13,
        pad=18,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"[图] 实测指标与文档声明对比 -> {output_path}")


def main():
    ap = argparse.ArgumentParser(description="统一验证脚本")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--device", default="0",
                    help="GPU 编号 (0-7 或 cpu; 默认 0)")
    ap.add_argument("--dataset", default="mixed", choices=["bccd", "txl", "mixed"],
                    help="数据集选择: bccd=仅BCCD, txl=仅TXL-PBC, mixed=混合 (默认 mixed)")
    ap.add_argument("--data", default=None,
                    help="数据集 yaml 路径 (指定后覆盖 --dataset)")
    ap.add_argument("--imgsz", type=int, default=640, help="验证输入尺寸")
    ap.add_argument("--iou", type=float, default=0.7, help="验证 NMS 的 IoU 阈值")
    ap.add_argument("--weights", default=None,
                    help="逗号分隔的权重路径; 不指定则评估内置 MODELS")
    ap.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True,
                    help="保存混淆矩阵/PR 曲线 (默认开启; 用 --no-plots 关闭)")
    ap.add_argument("--project", default=RESULTS_DIR, help="结果输出目录 (混淆矩阵/PR 曲线)")
    args = ap.parse_args()

    # 参数范围保护: IoU 越界时静默夹取到 [0, 1]
    args.iou = max(0.0, min(1.0, args.iou))

    # 解析数据集 yaml (--data 优先, 否则按 --dataset 选择, 默认 mixed)
    if args.data:
        data_yaml = args.data
    else:
        data_yaml = ensure_dataset_ready(args.dataset)
    args.data = data_yaml

    if args.weights:
        models = []
        for p in args.weights.split(","):
            p = p.strip()
            if p:
                models.append((os.path.basename(p), p))
    else:
        models = list(MODELS.items())

    document_metrics = load_document_metrics()
    records = []
    print(f"评估集: {args.split}, 数据集: {args.dataset} ({args.data}), 设备: {args.device}\n")
    for name, path in models:
        if not os.path.exists(path):
            print(f"⚠️  跳过 {name}: 找不到 {path}")
            continue
        m = YOLO(path)
        r = m.val(data=args.data, split=args.split, device=args.device,
                  imgsz=args.imgsz, iou=args.iou,
                  project=args.project, name=f"{name}_{args.split}", exist_ok=True,
                  plots=args.plots, save=False, verbose=False)
        b = r.box
        names = getattr(r, "names", None) or {i: n for i, n in enumerate(NAMES)}
        records.append({
            "name": name,
            "precision": float(b.mp),
            "map50": float(b.map50),
        })
        print(f"[{name:<9}] mAP@0.5={b.map50:.4f}  mAP@0.5:0.95={b.map:.4f}")
        # 文档要求: 按类别与 ALL 分别输出 Precision / Recall / mAP@0.5
        print(f"           {'类别':<10} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10}")
        for i in range(b.nc):
            cls_name = names.get(i, str(i)) if isinstance(names, dict) else names[i]
            print(f"           {cls_name:<10} {b.p[i]:>10.4f} {b.r[i]:>10.4f} {b.ap50[i]:>10.4f}")
        print(f"           {'ALL':<10} {b.mp:>10.4f} {b.mr:>10.4f} {b.map50:>10.4f}")
        if args.plots:
            print(f"           [图] 混淆矩阵/PR 曲线 -> {os.path.join(args.project, f'{name}_{args.split}')}")
        print()

    save_accuracy_comparison(records, args, document_metrics)

    # 清理 ultralytics 校验时自动创建的空 runs 目录, 保持工作区干净
    shutil.rmtree(os.path.join(HERE, "runs"), ignore_errors=True)


if __name__ == "__main__":
    main()
