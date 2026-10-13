# YOLO26n 血细胞智能检测系统

## 交接（2026-10-13，明日继续）

- **工作分支：**`cell-26`；阶段提交：`6b45c41`。今天的代码与结果说明已推送。继续工作时留在此分支，不切换或修改其他分支。
- **数据集：**现成的 `dataset1/mixed_blood_cell.yaml`，train/val/test 为 1116/310/198 张。不要解压、重建、改写或提交 `dataset1/`。
- **已完成：**按 `doc.txt` 的消融要求训练并在同一 mixed test 集评估 YOLO26n baseline、focal、cbam、dams、cf 五组；Focal Loss 已适配 YOLO26 的 E2E loss。全组配置为 imgsz=640、Adam、lr0=0.001、batch=16、最多 100 epochs、全网微调、patience=20、Mosaic=0.5、close-mosaic=15、HSV-S/V=0.3/0.2、seed=0。
- **测试集 ALL 指标（P / R / mAP@0.5 / mAP@0.5:0.95）：**baseline `0.9112 / 0.9346 / 0.9669 / 0.8016`；focal `0.8892 / 0.9140 / 0.9529 / 0.8051`；cbam `0.9197 / 0.9351 / 0.9667 / 0.7984`；dams `0.9224 / 0.9378 / 0.9700 / 0.8045`；cf `0.9046 / 0.9071 / 0.9519 / 0.7960`。当前最佳为 **dams**；CF 尚未优于 baseline，不能把其他变体结果称作 CF 结果。
- **本机忽略产物：**五组权重位于 `weights/yolo26n_mixed.pt`、`weights/yolo26n_ab_{focal,cbam,dams,cf}.pt`；训练曲线 CSV 位于对应的 `results/*_train/results.csv`。这些路径受 `.gitignore` 忽略，**不要 `git add -f`，也不要提交权重、CSV、数据集或临时输出**。上面完整的 RBC/WBC 指标表见下方“YOLO26n 超参数实验”。
- **明日优先事项：**仅继续改进 `cf`，从当前 CF 权重 `weights/yolo26n_ab_cf.pt` 出发，先用验证集测试 Focal α（建议 0.50、0.75；γ=2.0），分别使用新的 `--out` 名称和 `--skip-test`，避免相互覆盖或用 test 选超参数；按验证集选择后，再对最终候选做一次统一 test 评估。比较 Focal 是否确实有助于 RBC/WBC 后，再决定是否调 γ 或模块组合。
- **复现实验：**`scripts/main.sh` 已改为只检查并读取现有 mixed YAML，不再解压或重建数据；它会重新训练全部五组并写入同名本地忽略产物，继续跑 CF 单项调参时不要无必要地重跑它。

基于 YOLO26n 的血细胞检测训练与演示系统，保留 CBAM、Focal Loss、DAMS-Net 等可选改进模块；历史 YOLOv8 实验文档见 `doc.txt`。

- 模型变体：`baseline`（纯 YOLO26n，默认）/ `focal`（YOLO26n + Focal Loss）/ `cbam`（YOLO26n + CBAM）/ `dams`（YOLO26n + DAMS-Net）/ `cf`（YOLO26n + CBAM + Focal Loss + DAMS-Net）
- 检测类别：`RBC` 红细胞 / `WBC` 白细胞（文档口径，仅两类）
- 功能：训练（`train.py`）、推理（`inf.py`）、验证（`eval.py`）、Web 演示（`app.py`）、摄像头实时检测（`camera.py`）、VOC→YOLO 转换（`convert_voc_to_yolo.py`）、模型轻量化（`lightweight.py`）、SDK 与 API（`sdk/`，REST: `POST /api/detect`）

## 环境准备

```bash
conda env create -f environment.yml   # 创建环境（环境名 cell）
conda activate cell
```

关键依赖：`ultralytics==8.4.130`（YOLO26 支持）、`torch==2.10.0+cu128`、`opencv-python`、`flask`。训练使用 `weights/yolo26n.pt` COCO 预训练权重。
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
| `main.sh` | 依次训练 YOLO26n baseline→focal→cbam→dams→cf，再验证与推理（混合数据集，默认超参） |
| `train.sh` | 训练 YOLO26n baseline（Adam、lr0=0.001、batch16、freeze10、100 轮、mixed） |
| `eval.sh` | 在混合测试集验证内置权重 |
| `inf.sh` | 在混合测试集用训练后的 YOLO26n 权重推理 |
| `app.sh` | 启动 Web 检测系统（端口 5000） |

```bash
bash scripts/main.sh
```

> 文档补齐项：`--variant dams`（DAMS-Net 消融）与 `--variant cf`（完整 CF-YOLOv8，含 DAMS-Net，权重名 `cf.pt`）；
> `eval.py --plots` 会把混淆矩阵/PR 曲线存到 `cell/results/`；`inf.py --benchmark` 输出 FPS；
> `convert_voc_to_yolo.py` 完成 VOC→YOLO 转换；`lightweight.py quantize/prune/distill` 做 INT8 量化、结构化剪枝与知识蒸馏（蒸馏复用 ultralytics 内置 `distill_model`，教师 CF-YOLOv8 → 剪枝学生）。
> `eval.py` 还会根据 `doc.txt` 中的 Precision 和 mAP@0.5 声明生成模型对比表图片，输出到 `cell/temp/accuracy_comparison_<dataset>_<split>.png`。

## 历史 YOLOv8 实验结果（mixed 2 类 · BCCD + TXL-PBC）

以下结果为原 YOLOv8 实验记录；当前 `train.py` 的默认基线已切换为 YOLO26n。

原 YOLOv8 结果来自此前的训练记录；当前 `scripts/main.sh` 会基于 YOLO26n 重新训练 5 个变体，不能复现此表中的旧数值。

训练配置：`Adam / lr0=0.001 / batch16 / freeze-epochs 10（先冻结后解冻）/ patience 10 / 100 轮 / 余弦退火`；Focal Loss `γ=2.0、α=0.25`（α-balanced：正样本权重 0.25、负样本 0.75，仅 cls 分支，box/DFL 不变）；CBAM（通道 1 输入 + 3 隐藏 + 1 输出 + 空间 7×7，置于 Conv 后、BN 前，reduction=8）；DAMS-Net（替换 Neck 的 Concat 固定融合，内容自适应门控加权）。

在 `dataset1/mixed_blood_cell` 测试集（198 张，RBC/WBC 两类）上的结果：

| 变体 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | RBC mAP@0.5 | WBC mAP@0.5 |
|---|---|---|---|---|---|---|
| baseline（纯 YOLOv8n） | 0.9633 | 0.7753 | 0.9037 | 0.9370 | 0.9369 | 0.9896 |
| focal（+Focal Loss） | 0.9481 | 0.7567 | 0.8794 | 0.9305 | 0.9071 | 0.9892 |
| cbam（+CBAM） | 0.9643 | 0.7479 | 0.9085 | 0.9344 | 0.9364 | 0.9921 |
| dams（+DAMS-Net） | 0.9653 | 0.7759 | 0.9144 | 0.9351 | 0.9378 | 0.9929 |
| cf（CF-YOLOv8） | 0.9562 | 0.7778 | 0.9067 | 0.9157 | 0.9222 | 0.9903 |

文档声称指标：`mAP@0.5 ≈ 0.947`、精确率 `89.7%`（目标 > 90%）。
CF-YOLOv8 实测 `mAP@0.5 = 0.9562（>0.947）`、`Precision = 0.9067（>90%，也 >89.7%）`，且 `mAP@0.5:0.95 = 0.7778` 为所有变体最高、反超 baseline（0.7753），均已达标。

> 说明：消融趋势（mAP@0.5）为 dams(0.9653) > cbam(0.9643) > baseline(0.9633) > cf(0.9562) > focal(0.9481)。
> CBAM、DAMS 单独加入均已超过 baseline；Focal Loss 仍是唯一拉低 mAP@0.5 的模块（压低了占多数的“易分”RBC 类），
> 而文档假设的“WBC 因长尾难学”在本数据集上不成立（WBC 已 ≈0.99）。
> 绝对指标达标依靠 2 类口径 + mixed 数据扩充；Focal 采用标准 α-balanced 形式（正样本 α=0.25、负样本 0.75）。

## YOLO26n 超参数实验（mixed 2 类 · BCCD + TXL-PBC）

本轮按 `doc.txt` 与 `scripts/main.sh` 重新完成五组 YOLO26n 消融：baseline、+Focal Loss、+CBAM、+DAMS-Net、CF（CBAM + Focal Loss + DAMS-Net）。所有变体使用相同的 mixed 数据集（1116 张训练图、310 张验证图、198 张测试图）、COCO 预训练权重和训练配置：全网微调、640 输入、Adam、lr0=0.001、batch=16、最多 100 轮、patience=20、Mosaic=0.5、HSV-S/V=0.3/0.2、seed=0。训练 checkpoint 按验证集 mAP@0.5 选优，以下为统一测试集评估；没有更改数据集。

Focal Loss 已适配 YOLO26 的端到端检测损失：通过 `E2ELoss` 分别包裹 one-to-many 与 one-to-one 检测分支，并只替换分类 BCE 项。

### ALL 测试指标

| YOLO26n 变体 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| baseline | 0.9112 | 0.9346 | 0.9669 | 0.8016 |
| +Focal Loss | 0.8892 | 0.9140 | 0.9529 | **0.8051** |
| +CBAM | 0.9197 | 0.9351 | 0.9667 | 0.7984 |
| +DAMS-Net | **0.9224** | **0.9378** | **0.9700** | 0.8045 |
| CF（CBAM + Focal + DAMS） | 0.9046 | 0.9071 | 0.9519 | 0.7960 |
| YOLOv8n 原 baseline（历史参考） | 0.9037 | 0.9370 | 0.9633 | 0.7753 |

### RBC / WBC 测试指标（Precision / Recall / mAP@0.5）

| YOLO26n 变体 | RBC | WBC |
|---|---|---|
| baseline | 0.8543 / 0.8692 / 0.9423 | 0.9681 / 1.0000 / 0.9915 |
| +Focal Loss | 0.8327 / 0.8429 / 0.9185 | 0.9457 / 0.9851 / 0.9873 |
| +CBAM | 0.8541 / 0.8733 / 0.9407 | 0.9853 / 0.9969 / 0.9928 |
| +DAMS-Net | 0.8595 / 0.8817 / 0.9469 | 0.9853 / 0.9940 / 0.9931 |
| CF（CBAM + Focal + DAMS） | 0.8383 / 0.8224 / 0.9140 | 0.9709 / 0.9918 / 0.9898 |

本轮最佳总体变体是 **+DAMS-Net**：相比 YOLO26n baseline，mAP@0.5 +0.0031、mAP@0.5:0.95 +0.0029、Precision +0.0112、Recall +0.0032；也高于历史 YOLOv8n baseline 的四项指标。**当前 CF 组合没有超过 baseline/DAMS**，Focal Loss 是组合中主要的性能回退因素；因此不能把 baseline 或 DAMS 的结果称为 CF 结果。后续将继续调优 CF 的 Focal Loss 权重，当前结果仅作为阶段性消融结论。

训练权重与逐轮 CSV 属于本地忽略产物（`cell/weights/`、`cell/results/`），不纳入 Git 提交。`scripts/main.sh` 使用独立 `yolo26n_ab_*` 名称，避免覆盖历史 YOLOv8 权重；要求本地已有 `dataset1/mixed_blood_cell.yaml`，不会解压或重建数据集。训练阶段可用 `--skip-test`，全部模型训练结束后再统一测试。

## Web 检测系统 (app.py)

**CLI 启动：**

```bash
python app.py                                 # 默认端口 5000
python app.py --port 8080                     # 指定端口
python app.py --device 6 --conf 0.3           # 指定 GPU 与置信度阈值
python app.py --weights weights/yolo26n_mixed.pt --device 0
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--host` | 监听地址 | `0.0.0.0` |
| `--port` | 端口 | `5000` |
| `--device` | GPU 编号（0-7 或 cpu） | `0` |
| `--weights` | 模型权重路径 | `weights/yolo26n_mixed.pt` |
| `--conf` | 置信度阈值 | `0.25` |
| `--debug` | 开启 Flask 调试模式 | 关闭 |

环境变量：`CELL_HOST` / `CELL_PORT` / `CELL_DEVICE` / `CELL_WEIGHTS` / `CELL_CONF` 可覆盖上述参数。

启动后浏览器打开 http://127.0.0.1:5000 即可使用。
网页内（左下角“设置”面板）支持：查看/切换权重、查看/切换推理设备（GPU/CPU，实时显示空闲显存）、
调整置信度阈值、查看/搜索/回放/清空检测历史记录。


## 摄像头实时检测 (camera.py)

环境：Python（先激活 conda 环境，在 `cell` 目录下执行，需在有摄像头与显示器的机器上运行）。

```bash
# 默认摄像头 0 + weights/yolo26n_mixed.pt，实时检测并按 q 退出
python camera.py
# 可指定摄像头、权重、设备与置信度阈值
python camera.py --camera 0 --weights weights/yolo26n_mixed.pt --device 0 --conf 0.25
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--camera` | 摄像头编号 | `0` |
| `--weights` | 模型权重路径 | `weights/yolo26n_mixed.pt` |
| `--device` | 推理设备（0-7 或 cpu） | `0` |
| `--conf` | 置信度阈值 | `0.25` |

说明：程序打开摄像头 0，加载 `weights/yolo26n_mixed.pt`，以置信度 0.25 逐帧检测并绘制检测框（检测框 + 类别标签 + 置信度），
实时输出各类细胞统计计数，按 `q` 退出。摄像头模式下已关闭逐帧日志并设置底层计算库线程数，退出时自动释放摄像头资源。


## 训练 / 推理 / 验证 (train.py / inf.py / eval.py)

### train.py —— 训练脚本

先激活 conda 环境，在 `cell` 目录下执行。默认使用 `weights/yolo26n.pt`，`--variant` 可选 `baseline` / `focal` / `cbam` / `dams` / `cf`。

```bash
python train.py --variant baseline --device 0 --epochs 100 --batch 16 --out yolo26n_mixed
python train.py --variant focal --device 0                     # YOLO26n + Focal Loss (消融)
python train.py --variant cbam --device 0                      # YOLO26n + CBAM
python train.py --variant baseline --dataset bccd --device 0   # 仅 BCCD 数据集
python train.py --variant baseline --dataset txl --device 0    # 仅 TXL-PBC 数据集
python train.py --variant baseline --freeze-epochs 10 --device 0
```

完整参数（默认值见括号）：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--variant` | 变体 | `baseline`（baseline/focal/cbam/dams/cf） |
| `--device` | GPU 编号 | `0`（0-7 或 cpu） |
| `--epochs` | 训练轮数 | `100` |
| `--batch` | Batch size | `16` |
| `--gamma` | Focal Loss γ | `2.0` |
| `--alpha` | Focal Loss α | `0.25` |
| `--freeze-epochs` | 冻结主干训练检测头的轮数 | `0`（全网微调） |
| `--freeze-layers` | 阶段 1 冻结的 backbone 层数 | `10` |
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
| `--mosaic` | Mosaic 增强概率（0 关闭） | `0.5` |
| `--close-mosaic` | 最后 N 轮关闭 Mosaic | `15` |
| `--degrees` | 随机旋转角度范围 | `180` |
| `--flipud` | 上下翻转概率 | `0.5` |
| `--fliplr` | 左右翻转概率 | `0.5` |
| `--hsv-h` | HSV-H 扰动 | `0.015` |
| `--hsv-s` | HSV-S 扰动 | `0.3` |
| `--hsv-v` | HSV-V 扰动 | `0.2` |
| `--patience` | 早停耐心轮数 | `20` |
| `--seed` | 随机种子 | `0` |
| `--no-deterministic` | 关闭确定性训练 | 关闭 |
| `--no-amp` | 关闭混合精度 AMP | 关闭 |
| `--no-plots` | 不保存训练曲线图 | 关闭 |

训练完成后 best 权重统一保存到 `weights/<variant>.pt`（或 `--out` 指定的文件）；当前实验权重使用独立名字（如 `yolo26n_mixed.pt`），不覆盖历史权重。

**消融调试要点**（以下五个旋钮均可单独命令行控制，无需改代码）：

| 旋钮 | 参数 | 默认值 | 文档要求 |
|---|---|---|---|
| Focal γ | `--gamma` | 2.0 | 2.0 |
| Focal α | `--alpha` | 0.25 | 0.25 |
| 数据集 | `--dataset bccd/txl/mixed` | mixed | BCCD |
| 全网微调 | `--freeze-epochs` | 0 | 不冻结 |

```bash
# 仅 BCCD 数据集
python train.py --variant baseline --dataset bccd --device 0
# 可选：冻结前 10 层训练检测头，再进行全网微调
python train.py --variant baseline --freeze-epochs 10 --freeze-layers 10 --device 0
```

### inf.py —— 推理脚本（单图或目录）

```bash
python inf.py --source a.jpg                                  # 单图
python inf.py --source dataset/BCCD_Dataset/test/images       # 目录
python inf.py --dataset mixed                                 # 不传 --source，自动用混合测试集
python inf.py --dataset bccd --weights weights/yolo26n_mixed.pt --conf 0.3 --save
python inf.py --source a.jpg --imgsz 640 --iou 0.7 --max-det 300 --save-txt --save-conf
```

完整参数（默认值见括号）：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--source` | 图片路径或目录 | 未指定时用 `--dataset` 对应的测试集 |
| `--dataset` | 数据集 | `mixed`（bccd/txl/mixed，仅用于默认测试图目录） |
| `--weights` | 权重路径 | `weights/yolo26n_mixed.pt` |
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
python eval.py --weights weights/yolo26n_mixed.pt,weights/cbam.pt --device 0
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
