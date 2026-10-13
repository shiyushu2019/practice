# YOLO26n SDK 与 API 接口文档

对应 `doc.txt` 交付物要求：SDK 开发包（提供 API 接口），输入血液显微图片，
输出检测框、细胞类别、置信度、细胞统计计数结果。

---

## 一、Python SDK（`cell/sdk/`）

SDK 封装在 `cell/sdk/__init__.py`，内部复用 `components.detector.CellDetector`。
导入时会自动把 `cell` 目录加入 `sys.path`，因此**任意工作目录**都能直接导入。

### 1. 快速开始

```python
from sdk import CellSDK

sdk = CellSDK(weights="weights/yolo26n_mixed.pt", device="0", conf=0.25)

res = sdk.detect("blood.jpg")
print(res.total)      # 总细胞数
print(res.counts)     # {'RBC': 18, 'WBC': 2}
for b in res.boxes:
    print(b.name, b.conf, b.xyxy)   # 类别名 / 置信度 / [x1, y1, x2, y2]
```

### 2. 主要接口

| 接口 | 说明 |
|---|---|
| `CellSDK(weights=None, device="0", conf=0.25)` | 构造并加载模型 |
| `sdk.detect(image_path, conf=None) -> DetectionResult` | 单图检测 |
| `sdk.detect_many(image_paths, conf=None) -> List[DetectionResult]` | 批量检测 |
| `sdk.model_info -> dict` | 模型信息（架构/类别/权重/设备/阈值） |

### 3. 返回结构

`DetectionResult`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `filename` | `str` | 图片文件名 |
| `total` | `int` | 检测到的细胞总数 |
| `counts` | `dict` | 各类计数，如 `{"RBC": 18, "WBC": 2}` |
| `boxes` | `List[DetectionBox]` | 检测框列表 |
| `avg_conf` | `float` | 平均置信度 |
| `cost_ms` | `float` | 单张推理耗时（毫秒） |

`DetectionBox`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `cls` | `int` | 类别序号（0=RBC, 1=WBC） |
| `name` | `str` | 类别名 |
| `conf` | `float` | 置信度 |
| `xyxy` | `List[float]` | 像素坐标 `[x1, y1, x2, y2]` |

---

## 二、REST API（`app.py`）

启动 Web 服务后即可调用：

```bash
python app.py --device 0 --weights weights/yolo26n_mixed.pt --port 5000
```

### 1. 单图检测（SDK 接口）`POST /api/detect`

输入一张血液显微图片，同步返回检测框、类别、置信度、计数。

**请求**（`multipart/form-data`）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image`（或 `file`） | file | 是 | 图片文件 |
| `conf` | float | 否 | 置信度阈值，默认 0.25（越界自动夹取） |

**响应**（`application/json`）：

```json
{
  "filename": "blood.jpg",
  "total": 20,
  "counts": {"RBC": 18, "WBC": 2},
  "avg_conf": 0.867,
  "cost_ms": 12.3,
  "boxes": [
    {"cls": 0, "name": "RBC", "conf": 0.912, "xyxy": [12.3, 34.5, 45.6, 67.8]}
  ]
}
```

**示例**：

```bash
curl -F "image=@blood.jpg" -F "conf=0.3" http://127.0.0.1:5000/api/detect
```

```python
import requests
r = requests.post("http://127.0.0.1:5000/api/detect",
                  files={"image": open("blood.jpg", "rb")},
                  data={"conf": "0.3"})
print(r.json())
```

### 2. 其他接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/model_info` | 当前模型信息 |
| GET | `/api/weights` | 列出 `weights/` 下可用权重 |
| POST | `/api/weights/switch` | 切换权重（`{"weights": "cbam.pt"}`） |
| GET | `/api/devices` | 列出 GPU/CPU 设备与空闲显存 |
| POST | `/api/device/switch` | 切换推理设备（`{"device": "1"}`） |
| POST | `/api/dashboard/start` | 批量上传图片开始异步检测 |
| GET | `/api/dashboard/status` | 查询批量任务进度与统计 |
| POST | `/api/dashboard/stop` | 停止批量任务 |
| GET | `/api/history` | 查询检测历史（支持 `keyword` 搜索） |
| POST | `/api/history/clear` | 清空检测历史 |
| GET | `/api/result/image/<filename>` | 获取可视化结果图 |

---

## 三、输出口径说明

- 类别固定为 2 类：`0=RBC`（红细胞）、`1=WBC`（白细胞）。
- `counts` 为各类细胞统计计数，`total` 为所有类之和。
- `xyxy` 为原图像素坐标（左上、右下），可直接用于在原图上绘制检测框。
