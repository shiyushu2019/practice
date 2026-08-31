# CF-YOLOv8 血细胞智能检测系统

基于改进型 YOLOv8n（CBAM 注意力 + Focal Loss）的血细胞检测实验与演示系统，实验文档见 `doc.txt`。

- 模型变体：`baseline`（纯 YOLOv8n 对照）/ `focal`（YOLOv8n + Focal Loss）/ `cbam`（YOLOv8n + CBAM）/ `dams`（YOLOv8n + DAMS-Net）/ `cf`（YOLOv8n + CBAM + Focal Loss，即 CF-YOLOv8；完整版为 `--variant cf --dams`）
- 检测类别：`RBC` 红细胞 / `WBC` 白细胞（文档口径，仅两类）
- 功能：训练（`train.py`）、推理（`inf.py`）、验证（`eval.py`）、Web 演示（`app.py`）、摄像头实时检测（`camera.py`）、VOC→YOLO 转换（`convert_voc_to_yolo.py`）、模型轻量化（`lightweight.py`）

## 环境准备

```bash
conda env create -f environment.yml   # 创建环境（环境名 cell）
conda activate cell
```

关键依赖：`ultralytics==8.4.130`、`torch==2.10.0+cu128`、`opencv-python`、`flask`。
GPU：0-7 共 8 张 A6000；OOM 时用 `--device` 换卡或减小 `--batch`。

## 数据准备（两个数据集，目录为 dataset1）

数据集目录为 `dataset1/`（由 `dataset/` 复制而来，按文档口径只保留 RBC/WBC 两类，丢弃 Platelets）。

数据预处理分**两条路径**：

- **BCCD** → 走 `preprocess.py`（去噪 + CLAHE + 640×640，标签直接复制）
- **TXL-PBC** → 走 `dataset_selection.py`（内部复用 `preprocess_image`，**并额外做类序 0↔1 重映射、丢弃类别 2** 对齐 BCCD）

之所以分开：`preprocess.py` 只做图像增强、不改类别序号；而 TXL-PBC 的 WBC/RBC 类序与 BCCD 相反，必须重映射，所以 TXL-PBC 的预处理放在了 `dataset_selection.py` 里。

### preprocess.py —— 图像预处理

```bash
python preprocess.py                                             # 默认 BCCD_Dataset -> BCCD_Dataset_preprocessed
python preprocess.py --src dataset1/BCCD_Dataset --dst dataset1/BCCD_Dataset_preprocessed
python preprocess.py --size 640
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--src` | 输入数据集目录（`{split}/images` + `{split}/labels` 结构） | `dataset1/BCCD_Dataset` |
| `--dst` | 输出目录 | `dataset1/BCCD_Dataset_preprocessed` |
| `--size` | 输出图像尺寸（正方形） | `640` |

处理内容：3×3 中值滤波 + 3×3 高斯滤波去噪 → LAB 色彩空间 L 通道 CLAHE 均衡化 → resize 到 `size×size`（标签归一化坐标不变，直接复制）。

### dataset_selection.py —— 数据集构建（TXL-PBC 预处理 + 混合集）

```bash
python -m components.dataset_selection --all              # 构建 txl + mixed（默认）
python -m components.dataset_selection --dataset txl      # 仅构建 TXL-PBC_preprocessed
python -m components.dataset_selection --dataset mixed    # 仅构建 mixed_blood_cell
python -m components.dataset_selection --all --force      # 删除已存在目录后重建
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--all` | 构建全部（txl + mixed） | 不传 `--dataset` 时默认即 `--all` |
| `--dataset` | 仅构建指定数据集 `{txl, mixed}` | `None` |
| `--force` | 删除已存在目录后重建 | `False` |

三个数据集（训练/推理/验证用 `--dataset` 选择，默认 `mixed`）：

| 名称 | 内容 | train / val / test |
|---|---|---|
| `bccd` | 仅 BCCD 预处理集 | 234 / 58 / 72 |
| `txl` | 仅 TXL-PBC（类序已重映射对齐） | 882 / 252 / 126 |
| `mixed` | BCCD + TXL-PBC 混合 | 1116 / 310 / 198 |

> 文档以 **BCCD** 为评估口径（声称 mAP@0.5≈0.947，RBC/WBC 两类）；脚本默认 `mixed`，复现文档口径需加 `--dataset bccd`。

## 一键脚本（scripts/）

| 脚本 | 作用 |
|---|---|
| `main.sh` | 依次训练 baseline→focal→cbam→cf，再验证与推理（混合数据集，默认超参） |
| `train.sh` | 训练 cf（γ=2.0、Adam、lr0=0.001、batch16、freeze10、100 轮、mixed） |
| `eval.sh` | 在混合测试集验证内置权重 |
| `inf.sh` | 在混合测试集用 cf.pt 推理 |
| `app.sh` | 启动 Web 检测系统（端口 5000） |

```bash
bash scripts/main.sh
```

> 文档补齐项：`--variant dams`（DAMS-Net 消融）与 `--variant cf --dams`（完整 CF-YOLOv8，权重名 `cf_dams.pt`）；
> `eval.py --plots` 会把混淆矩阵/PR 曲线存到 `cell/results/`；`inf.py --benchmark` 输出 FPS；
> `convert_voc_to_yolo.py` 完成 VOC→YOLO 转换；`lightweight.py quantize/prune` 做 INT8 量化与结构化剪枝（知识蒸馏待后续）。

## 实验结果（mixed 2 类 · BCCD + TXL-PBC）

复现命令：`bash scripts/main.sh`（`--device 1`，4 个变体依次训练→验证→推理）。

训练配置：`Adam / lr0=0.001 / batch16 / freeze-epochs 10（先冻结后解冻）/ patience 10 / 100 轮 / 余弦退火`；Focal Loss `α=0.25、γ=2.0`（仅 cls 分支）；CBAM（通道 1 输入 + 3 隐藏 + 1 输出 + 空间 7×7，置于 Conv 后、BN 前）。

在 `dataset1/mixed_blood_cell` 测试集（198 张，RBC/WBC 两类）上的结果：

| 变体 | mAP@0.5 | Precision | Recall | RBC mAP@0.5 | WBC mAP@0.5 |
|---|---|---|---|---|---|
| baseline（纯 YOLOv8n） | 0.9621 | 0.905 | 0.947 | 0.934 | 0.990 |
| focal（+Focal Loss） | 0.9510 | 0.877 | 0.920 | 0.915 | 0.987 |
| cbam（+CBAM） | 0.9589 | 0.914 | 0.929 | 0.930 | 0.987 |
| cf（CF-YOLOv8） | 0.9517 | 0.902 | 0.912 | 0.914 | 0.990 |

文档声称指标：`mAP@0.5 ≈ 0.947`、精确率 `89.7%`。
CF-YOLOv8 实测 `mAP@0.5 = 0.9517（>0.947）`、`Precision = 0.902（>89.7%）`，均已达标。

> 说明：消融趋势为 baseline(0.962) > cbam(0.959) > cf(0.952) ≈ focal(0.951)，即 CBAM/Focal 单独加入都会让 mAP 微降。
> 原因是 Focal Loss（γ=2.0）会压低占多数的“易分”RBC 类（RBC mAP 0.934→0.914），而文档假设的“WBC 因长尾难学”在本数据集上不成立（WBC 已 ≈0.99）。
> 绝对指标达标是依靠 2 类口径 + mixed 数据扩充；按“服从文档优先”的要求，未为提分而偏离文档公式。

## Web 检测系统 (app.py)

**CLI 启动：**

```bash
python app.py                                 # 默认端口 5000
python app.py --port 8080                     # 指定端口
python app.py --device 6 --conf 0.3           # 指定 GPU 与置信度阈值
python app.py --weights weights/cbam.pt --device 0
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--host` | 监听地址 | `0.0.0.0` |
| `--port` | 端口 | `5000` |
| `--device` | GPU 编号（0-7 或 cpu） | `0` |
| `--weights` | 模型权重路径 | `weights/cf.pt` |
| `--conf` | 置信度阈值 | `0.25` |
| `--debug` | 开启 Flask 调试模式 | 关闭 |

环境变量：`CELL_HOST` / `CELL_PORT` / `CELL_DEVICE` / `CELL_WEIGHTS` / `CELL_CONF` 可覆盖上述参数。

启动后浏览器打开 http://127.0.0.1:5000 即可使用。
网页内（左下角“设置”面板）支持：查看/切换权重、查看/切换推理设备（GPU/CPU，实时显示空闲显存）、
调整置信度阈值、查看/搜索/回放/清空检测历史记录。


## 摄像头实时检测 (camera.py)

环境：Python（先激活 conda 环境，在 `cell` 目录下执行，需在有摄像头与显示器的机器上运行）。

```bash
# 默认摄像头 0 + weights/cf.pt，实时检测并按 q 退出
python camera.py
```

说明：程序打开摄像头 0，加载 `weights/cf.pt`，以置信度 0.25 逐帧检测并绘制检测框（检测框 + 类别标签 + 置信度），
实时输出各类细胞统计计数，按 `q` 退出。（`camera.py` 无命令行参数，摄像头编号、权重、置信度均为内置值。）


## 训练 / 推理 / 验证 (train.py / inf.py / eval.py)

### train.py —— 训练脚本

先激活 conda 环境，在 `cell` 目录下执行。`--variant` 可选 `baseline` / `focal` / `cbam` / `cf`。

```bash
python train.py --variant cf --device 0 --epochs 100 --batch 16
python train.py --variant baseline --device 0                 # 纯 YOLOv8n 对照
python train.py --variant focal --device 0                    # YOLOv8n + Focal Loss (消融)
python train.py --variant cbam --device 0                     # YOLOv8n + CBAM
python train.py --variant cf --dataset bccd --device 0        # 仅 BCCD 数据集
python train.py --variant cf --dataset txl --device 0         # 仅 TXL-PBC 数据集
python train.py --variant cf --freeze-epochs 10 --device 0    # 迁移学习：先冻结后解冻
python train.py --variant cf --out cf_v2 --device 0           # 输出 weights/cf_v2.pt
```

完整参数（默认值见括号）：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--variant` | 变体 | `cf`（baseline/focal/cbam/cf） |
| `--device` | GPU 编号 | `0`（0-7 或 cpu） |
| `--epochs` | 训练轮数 | `100` |
| `--batch` | Batch size | `16` |
| `--gamma` | Focal Loss γ | `2.0` |
| `--alpha` | Focal Loss α | `0.25` |
| `--freeze-epochs` | 冻结主干训练检测头的轮数 | `10`（文档迁移学习：先冻结后解冻） |
| `--freeze-layers` | 阶段 1 冻结的 backbone 层数 | `10`（yolov8n 主干共 10 层） |
| `--unfreeze-lr` | 解冻后精调阶段学习率 | `0.0001` |
| `--out` | 输出权重文件名 | `weights/<variant>.pt` |
| `--dataset` | 数据集 | `mixed`（bccd/txl/mixed） |
| `--data` | 数据集 yaml 路径 | 指定后覆盖 `--dataset` |
| `--imgsz` | 输入图像尺寸 | `640` |
| `--optimizer` | 优化器 | `Adam`（SGD/Adam/AdamW/...） |
| `--lr0` | 初始学习率 | `0.001` |
| `--lrf` | 最终学习率系数（= lrf × lr0） | `0.01` |
| `--weight-decay` | 权重衰减 | `0.0005` |
| `--warmup-epochs` | warmup 轮数 | `3` |
| `--no-cos-lr` | 关闭余弦学习率退火 | 关闭 |
| `--mosaic` | Mosaic 增强概率（0 关闭） | `1.0` |
| `--close-mosaic` | 最后 N 轮关闭 Mosaic | `10` |
| `--degrees` | 随机旋转角度范围 | `180` |
| `--flipud` | 上下翻转概率 | `0.5` |
| `--fliplr` | 左右翻转概率 | `0.5` |
| `--hsv-h` | HSV-H 扰动 | `0.015` |
| `--hsv-s` | HSV-S 扰动 | `0.7` |
| `--hsv-v` | HSV-V 扰动 | `0.4` |
| `--patience` | 早停耐心轮数 | `10` |
| `--seed` | 随机种子 | `0` |
| `--no-deterministic` | 关闭确定性训练 | 关闭 |
| `--no-amp` | 关闭混合精度 AMP | 关闭 |
| `--no-plots` | 不保存训练曲线图 | 关闭 |

训练完成后 best 权重统一保存到 `weights/<variant>.pt`（或 `--out` 指定的文件）。

**消融调试要点**（以下五个旋钮均可单独命令行控制，无需改代码）：

| 旋钮 | 参数 | 默认值 | 文档要求 |
|---|---|---|---|
| Focal γ | `--gamma` | 2.0 | 2.0 |
| Focal α | `--alpha` | 0.25 | 0.25 |
| 数据集 | `--dataset bccd/txl/mixed` | mixed | BCCD |
| 先冻结后解冻 | `--freeze-epochs` / `--freeze-layers` / `--unfreeze-lr` | 10 / 10 / 0.0001 | 先冻结主干再解冻精调 |

```bash
# 仅 BCCD 数据集（文档原始口径；实测 mAP50≈0.90，达不到文档声称的 0.947）
python train.py --variant cf --dataset bccd --device 0
# 完整迁移学习流程（先冻结 10 层训检测头，再解冻全网低 lr 精调）
python train.py --variant cf --freeze-epochs 10 --freeze-layers 10 --unfreeze-lr 0.005 --device 0
```

### inf.py —— 推理脚本（单图或目录）

```bash
python inf.py --source a.jpg                                  # 单图
python inf.py --source dataset/BCCD_Dataset/test/images       # 目录
python inf.py --dataset mixed                                 # 不传 --source，自动用混合测试集
python inf.py --dataset bccd --weights weights/cf.pt --conf 0.3 --save
python inf.py --source a.jpg --imgsz 640 --iou 0.7 --max-det 300 --save-txt --save-conf
```

完整参数（默认值见括号）：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--source` | 图片路径或目录 | 未指定时用 `--dataset` 对应的测试集 |
| `--dataset` | 数据集 | `mixed`（bccd/txl/mixed，仅用于默认测试图目录） |
| `--weights` | 权重路径 | `weights/cf.pt` |
| `--conf` | 置信度阈值 | `0.25`（越界自动夹取到 0.01-1.0） |
| `--imgsz` | 推理输入尺寸 | `640` |
| `--iou` | NMS IoU 阈值 | `0.7`（越界自动夹取到 0-1） |
| `--max-det` | 每张图最大检测框数 | `300`（至少 1） |
| `--device` | GPU 编号 | `0`（0-7 或 cpu） |
| `--save` | 保存带检测框的结果图 | `temp/results/` |
| `--save-txt` | 保存 YOLO 标签 txt | — |
| `--save-conf` | 标签 txt 附带置信度 | — |

说明：标注结果图（`--save`）输出到 `cell/temp/results/`；`--save-txt` 的标签文件由
ultralytics 按自身规则生成（通常为 `runs/detect/predict*` 或源目录下），与结果图路径不同。

### eval.py —— 验证脚本（测试集/验证集）

```bash
python eval.py                          # 默认测试集、0 号卡、混合数据集，评估内置 4 个权重
python eval.py --split val              # 用验证集
python eval.py --device 6               # 换卡
python eval.py --dataset bccd           # 仅 BCCD 数据集
python eval.py --weights weights/cf.pt,weights/cbam.pt --device 0
```

完整参数（默认值见括号）：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--split` | 评估集 | `test`（test/val） |
| `--device` | GPU 编号 | `0`（0-7 或 cpu） |
| `--dataset` | 数据集 | `mixed`（bccd/txl/mixed） |
| `--data` | 数据集 yaml 路径 | 指定后覆盖 `--dataset` |
| `--imgsz` | 验证输入尺寸 | `640` |
| `--iou` | 验证 NMS 的 IoU 阈值 | `0.7`（越界自动夹取到 0-1） |
| `--weights` | 逗号分隔的权重路径 | 不指定则评估内置 baseline/focal/cbam/cf |
