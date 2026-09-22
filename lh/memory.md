# 项目复现记录

## 1. 项目目标

本项目研究人脸识别模型特征的可逆性/隐私泄露风险，整体攻击链为：

```text
输入人脸
  -> victim ResNet50
  -> ResNet 特征
  -> NewFE 映射到 AE 的 8192 维空间
  -> AE Inversion 解码器
  -> 重建人脸
```

三个核心训练阶段：

1. `resnetface.py`：训练 FaceScrub 人脸识别 victim 模型（ResNet50 特征提取器 + 分类头）。
2. `aeface.py`：训练独立的图像自编码/反演模型（卷积特征提取器 + 8192 维归一化特征 + 反卷积解码器）。
3. `nattackface.py`：冻结 victim、AE 编码器和 AE 解码器，只训练 `NewFE`，使 ResNet 特征映射到 AE 可解码的 8192 维空间。

`testnattackface.py` 用于加载模型并导出重建图；`main.py` 用于量化和压缩实验，不是新的训练阶段。

## 2. 数据集

当前本地数据目录：

```text
/data1/dataset/jssnu005/sys/lh/FaceScrub
```

实际检查结果：

- 图像总数：35,337
- 类别数：531
- 使用固定随机种子：`9970`

注意：原始代码注释中有 `class_num = 530`，但当前 `ImageFolder` 实际发现 531 个类别。ResNet 代码已经改为：

```python
class_num = len(train_source.classes)
```

因此当前分类头输出为 531 类。

## 3. 已完成的 ResNet 复现

`resnetface.py` 已改为使用本地数据目录、CUDA/CPU 自动选择、独立 checkpoint 目录和固定随机划分：

```text
V_train：30,288 张（6/7）
V_test ： 5,049 张（1/7）
```

划分方式：

```python
split_generator = torch.Generator().manual_seed(9970)
permutation = torch.randperm(ds_len, generator=split_generator).tolist()
train_ds = Subset(train_source, permutation[:ds_len * 6 // 7])
test_ds = Subset(test_source, permutation[ds_len * 6 // 7:])
```

训练和测试使用相同的样本索引，但训练集使用 `RandomHorizontalFlip()`，测试集不使用随机增强。

已完成运行记录：

- 训练轮数：48
- 训练集准确率：约 99.99%
- 测试集准确率：约 86.79%（epoch 48）
- 最佳记录附近测试准确率：约 86.81%（epoch 44）
- 未发现 Traceback、CUDA OOM、NaN 或异常退出
- checkpoint 保存为：

```text
./checkpoints/resnetface/feature_extractor_4.pkl
...
./checkpoints/resnetface/feature_extractor_48.pkl
./checkpoints/resnetface/cls_4.pkl
...
./checkpoints/resnetface/cls_48.pkl
```

## 4. 当前统一的数据划分方案

为保留已经训练好的 ResNet，同时避免后续 AE/NewFE 训练与 ResNet 训练数据重叠，四个训练/评估脚本现在统一使用同一个 `seed=9970` 生成的 permutation：

```text
FaceScrub：35,337
├── victim_train：30,288（前 6/7）
│   └── 已用于训练 ResNet；只作为后续 private 评估集
└── victim_test：5,049（后 1/7，ResNet 未见）
    ├── attacker_train：4,327（victim_test 的前 6/7）
    │   └── 用于训练 AE 和 NewFE
    └── attacker_test：722（victim_test 的后 1/7）
        └── 用于最终攻击/重建测试
```

集合关系已经验证为两两不重叠：

```text
victim_train ∩ attacker_train = ∅
victim_train ∩ attacker_test  = ∅
attacker_train ∩ attacker_test = ∅
```

这不是 `74fa685` 原始 AE 的 1/2 `priv_ds` + `aux_ds` 划分，而是当前在不重跑 ResNet 的前提下采用的无重叠后续训练方案。

## 5. 已完成的 AE 数据构造改造

`aeface.py` 已改为：

- 使用本地 FaceScrub 路径；
- 使用 `seed = 9970`；
- 使用与 `resnetface.py` 完全相同的 `torch.randperm`；
- 只从 ResNet 未见的 `victim_test` 中取 `attacker_train` 训练 AE；
- `test_ds` 使用 `attacker_test`；
- `priv_ds` 使用 `victim_train`，只用于 private 评估；
- 训练输入 resize 为 `128 x 128`；
- 训练集使用随机水平翻转；
- 测试集使用不带随机增强的确定性 transform；
- checkpoint 和图像输出保存到：

```text
./checkpoints/aeface/
```

当前 AE 数据集合定义为：

```text
train_ds：attacker_train，4,327 张
test_ds ：attacker_test，722 张
priv_ds ：victim_train，30,288 张
```

AE 模型结构：

```text
128x128 人脸
  -> 5 层卷积/池化 FeatureExtracter
  -> 8192 维特征
  -> F.normalize
  -> 6 层反卷积 Inversion
  -> 128x128 重建图
```

训练损失为 `SSIMLoss`，同时更新 AE 特征提取器和反演器。

已验证：

- `aeface.py` 可以通过 `python -m py_compile aeface.py`；
- 数据构造得到 30,288/5,049 的划分；
- 训练集和测试集的图像尺寸均为 `3 x 128 x 128`。

## 6. 后续脚本状态

### `nattackface.py`

- 已改为使用本地 FaceScrub 路径；
- 已加载 `./checkpoints/resnetface/feature_extractor_48.pkl` 和 `./checkpoints/aeface/` 下的 AE 权重；
- 已使用 `attacker_train` 训练 `NewFE`，`attacker_test` 测试，`victim_train` 做 private 评估。

### `testnattackface.py`

- 已改为使用同一套固定索引和本地 checkpoint；
- 会分别导出 attacker train、attacker test 和 victim train/private 的重建样例。

### `main.py`

- 按要求暂未修改，仍保留原版数据路径和划分；
- 量化实验暂时不能直接复用当前本地 checkpoint，待前三个阶段完成后再单独处理。

## 7. 推荐的后续顺序

1. 如果后台正在运行的是旧版 `df764a2` AE，应停止该进程；它使用了错误的 AE 数据划分。
2. 按当前 `aeface.py` 重新训练 AE，确认 `attacker_train`/`attacker_test` SSIM 和 checkpoint 正常生成。
3. 运行 `nattackface.py`，只更新 `NewFE`，冻结 victim/AE 模型。
4. 运行 `testnattackface.py`，分别导出三组重建样例。
5. 最后再决定是否修改并运行 `main.py` 做量化比较。

## 8. 交接运行说明

### 是否需要重跑 ResNet

不需要为了当前划分方案重跑 ResNet。`resnetface.py` 已经使用：

```text
seed=9970
victim_train=30,288
victim_test=5,049
```

后续脚本只是复用同一个 permutation，并从 `victim_test` 中继续拆分 `attacker_train` 和 `attacker_test`。如果希望从头完整重跑，也可以重新执行 `resnetface.py`；它的输出目录和 AE 不冲突：

```text
resnetface.py -> ./checkpoints/resnetface/
aeface.py     -> ./checkpoints/aeface/
```

### 推荐启动方式

如果不重跑 ResNet：

```bash
cd /data1/code/jssnu005/sys/practice/lh
python aeface.py > aeface.log 2>&1
# AE 完成后
python nattackface.py > nattackface.log 2>&1
# NewFE 完成后
python testnattackface.py > testnattackface.log 2>&1
```

如果想完整重跑，可以同时启动 `resnetface.py` 和 `aeface.py`，但两者会竞争同一块 GPU；显存不足时应改为串行。无论是否并行，必须等两者完成后再启动 `nattackface.py`。

### 阶段完成标志

启动 `nattackface.py` 前确认：

```text
./checkpoints/resnetface/feature_extractor_48.pkl
./checkpoints/aeface/feature_extractor_100.pkl
./checkpoints/aeface/myinversion_100.pkl
```

启动 `testnattackface.py` 前再确认：

```text
./checkpoints/nattackface/newfe_100.pkl
```

### 当前脚本输出目录

```text
./checkpoints/resnetface/
./checkpoints/aeface/
./checkpoints/nattackface/
./checkpoints/testnattackface/
```

`testnattackface.py` 会将样例输出到：

```text
./checkpoints/testnattackface/train/{input,output}/
./checkpoints/testnattackface/test/{input,output}/
./checkpoints/testnattackface/priv/{input,output}/
```

其中：

- `train` = `attacker_train`；
- `test` = `attacker_test`；
- `priv` = `victim_train`。

### 重要的历史版本区别

- `74fa685` 是导师原版；原版 `aeface.py` 使用全量数据 1/2 `priv_ds` + 1/2 `aux_ds`，与原版 ResNet 的独立随机划分可能重叠。
- `df764a2` 是曾经的兼容版本，曾把 AE 改成全量数据 6:1；这既不是导师原版 AE 划分，也没有解决与 victim 的重叠问题。
- 当前工作区的后续方案是经过重新确定的无重叠方案：复用 ResNet permutation，并只从 ResNet 未见数据训练 AE/NewFE。
- 因此当前方案的目标是“保留已有 ResNet、避免后续训练数据重叠”，不是逐字节复现 `74fa685` 的原始数据协议。

### 验证状态

当前 `resnetface.py`、`aeface.py`、`nattackface.py`、`testnattackface.py` 均已通过：

```bash
python -m py_compile resnetface.py aeface.py nattackface.py testnattackface.py
```

数据划分已通过脚本验证：

```text
total=35,337
victim_train=30,288
victim_test=5,049
attacker_train=4,327
attacker_test=722
```

在本次交接记录时，代码层面的 checkpoint 依赖已统一；是否已经生成 AE/NewFE checkpoint，应以文件是否存在和对应日志是否显示完成为准，不要只根据脚本改动判断。
