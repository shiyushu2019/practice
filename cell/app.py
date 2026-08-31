# -*- coding: utf-8 -*-
"""CF-YOLOv8 血细胞智能检测系统 (Flask Dashboard)。

参照 web_app_example 的架构与 UI 风格重写。

用法:
    python app.py                                  # 默认端口 5000
    python app.py --port 8080                      # 指定端口
    python app.py --host 0.0.0.0 --port 8080       # 指定监听地址与端口
    python app.py --device 6 --conf 0.3            # 指定推理设备与置信度阈值
"""
import argparse
import base64
import logging
import os
import shutil
import sys
import threading
import time
import uuid

import torch

from flask import Flask, render_template, request, jsonify, send_file

from components.detector import CellDetector

HERE = os.path.dirname(os.path.abspath(__file__))
# 兼容旧权重: 旧 checkpoint 里 focal_loss/cbam 是顶层模块, 反序列化时需能 import 到它们
sys.path.insert(0, os.path.join(HERE, "components"))

# 默认配置（可被命令行参数或环境变量覆盖）
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "cf.pt")
DEFAULT_CONF = 0.25
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000
WEIGHTS_DIR = os.path.join(HERE, "weights")

# UI 运行时产生的临时文件统一放在 cell/temp 下
TEMP_DIR = os.path.join(HERE, "temp")
UPLOAD_DIR = os.path.join(TEMP_DIR, "uploads")
CACHE_DIR = os.path.join(TEMP_DIR, "results")
STATIC_DIR = os.path.join(HERE, "components", "static")
LOG_FILE = os.path.join(TEMP_DIR, "app.log")

logger = logging.getLogger("cell.dashboard")


def parse_args():
    parser = argparse.ArgumentParser(description="CF-YOLOv8 血细胞智能检测系统")
    parser.add_argument("--host", default=os.environ.get("CELL_HOST", DEFAULT_HOST),
                        help=f"监听地址（默认 {DEFAULT_HOST}，可用环境变量 CELL_HOST 覆盖）")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CELL_PORT", DEFAULT_PORT)),
                        help=f"监听端口（默认 {DEFAULT_PORT}，可用环境变量 CELL_PORT 覆盖）")
    parser.add_argument("--device", default=os.environ.get("CELL_DEVICE", "0"),
                        help="推理设备（如 0-7 或 cpu，默认 0）")
    parser.add_argument("--weights", default=os.environ.get("CELL_WEIGHTS", DEFAULT_WEIGHTS),
                        help=f"模型权重路径（默认 {DEFAULT_WEIGHTS}）")
    parser.add_argument("--conf", type=float, default=float(os.environ.get("CELL_CONF", DEFAULT_CONF)),
                        help=f"默认置信度阈值（默认 {DEFAULT_CONF}）")
    parser.add_argument("--debug", action="store_true", help="开启 Flask 调试模式")
    return parser.parse_args()


app = Flask(
    __name__,
    template_folder=os.path.join(HERE, "components", "templates"),
    static_folder=STATIC_DIR,
)
app.secret_key = os.environ.get("CELL_SECRET_KEY", "cell-dashboard-secret-key-2026")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 运行时状态：detector / model_info 在 main() 中初始化
detector = None
model_info = {}

history_cache = []
cache_lock = threading.Lock()

task_state = {
    "running": False,
    "stop_requested": False,
    "files": [],
    "current_idx": 0,
    "conf": DEFAULT_CONF,
    "stats": {
        "success": 0, "fail": 0, "total": 0,
        "total_cells": 0, "total_conf": 0.0, "avg_conf": 0.0,
        "total_time": 0.0, "avg_time": 0.0,
        "rbc": 0, "wbc": 0,
    },
    "latest_result": {},
    "logs": [],
}
state_lock = threading.Lock()
model_lock = threading.Lock()


def process_image(filepath, conf=0.25):
    """对单张图像执行检测并写入历史缓存。"""
    with model_lock:
        result = detector.detect({"image_path": filepath, "conf": conf})
    if result is None:
        return None

    record_id = str(uuid.uuid4())
    orig_name = f"{record_id}_orig.jpg"
    result_name = f"{record_id}_result.jpg"

    with open(os.path.join(CACHE_DIR, orig_name), "wb") as f:
        f.write(base64.b64decode(result["original_b64"]))
    with open(os.path.join(CACHE_DIR, result_name), "wb") as f:
        f.write(base64.b64decode(result["result_b64"]))

    record = {
        "id": record_id,
        "filename": result["filename"],
        "orig_url": f"/api/result/image/{orig_name}",
        "result_url": f"/api/result/image/{result_name}",
        "counts": result["counts"],
        "total": result["total"],
        "avg_conf": result["avg_conf"],
        "report": result["report"],
        "timestamp": time.strftime('%H:%M:%S'),
    }

    with cache_lock:
        history_cache.insert(0, record)
        if len(history_cache) > 100:
            history_cache.pop()

    result["record"] = record
    return result


def processing_loop():
    global task_state
    while True:
        with state_lock:
            if (not task_state["running"] or task_state["stop_requested"]
                    or not (task_state["files"] and task_state["current_idx"] < len(task_state["files"]))):
                task_state["running"] = False
                break
            filepath = task_state["files"][task_state["current_idx"]]
            filename = os.path.basename(filepath)
            conf = task_state.get("conf", 0.25)
            task_state["stats"]["total"] = len(task_state["files"])

        start_proc_time = time.time()
        try:
            result = process_image(filepath, conf)
            cost = time.time() - start_proc_time
            with state_lock:
                if result:
                    task_state["stats"]["success"] += 1
                    task_state["stats"]["total_cells"] += result["total"]
                    # 每图只返回平均置信度, 乘以细胞数还原该图所有框置信度之和,
                    # 再除以总细胞数, 得到按框加权的真实全局平均置信度。
                    task_state["stats"]["total_conf"] += result["avg_conf"] * result["total"]
                    task_state["stats"]["avg_conf"] = (
                        task_state["stats"]["total_conf"] / task_state["stats"]["total_cells"]
                        if task_state["stats"]["total_cells"] else 0.0
                    )
                    task_state["stats"]["rbc"] += result["counts"]["RBC"]
                    task_state["stats"]["wbc"] += result["counts"]["WBC"]
                    task_state["latest_result"] = result
                    task_state["logs"].append(
                        f"[{time.strftime('%H:%M:%S')}] Success: {filename} ({cost:.2f}s, {result['total']} cells)"
                    )
                else:
                    task_state["stats"]["fail"] += 1
                    task_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] Failed: {filename} cannot read")
                task_state["stats"]["total_time"] += cost
                task_state["stats"]["avg_time"] = (
                    task_state["stats"]["total_time"] / (task_state["stats"]["success"] + task_state["stats"]["fail"])
                )
                task_state["current_idx"] += 1
                if len(task_state["logs"]) > 100:
                    task_state["logs"] = task_state["logs"][-100:]
        except Exception as e:
            with state_lock:
                task_state["stats"]["fail"] += 1
                task_state["stats"]["total_time"] += time.time() - start_proc_time
                task_state["stats"]["avg_time"] = (
                    task_state["stats"]["total_time"] / (task_state["stats"]["success"] + task_state["stats"]["fail"])
                )
                task_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] Error {filename}: {str(e)}")
                task_state["current_idx"] += 1
                if len(task_state["logs"]) > 100:
                    task_state["logs"] = task_state["logs"][-100:]

    with state_lock:
        task_state["running"] = False
        task_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] Task finished")


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/model_info')
def get_model_info():
    return jsonify({
        "arch": model_info.get("arch"),
        "classes": model_info.get("classes"),
        "classes_cn": model_info.get("classes_cn"),
        "weights": model_info.get("weights"),
        "device": model_info.get("device"),
        "conf": model_info.get("conf"),
    })


def _free_model():
    """释放当前模型占用的显存，避免切换时显存叠加导致 OOM。"""
    old = detector.model
    detector.model = None
    if old is not None:
        del old
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def list_available_weights():
    """扫描 weights 目录下的所有 .pt 权重文件。"""
    items = []
    if os.path.isdir(WEIGHTS_DIR):
        for name in sorted(os.listdir(WEIGHTS_DIR)):
            if not name.endswith(".pt"):
                continue
            path = os.path.join(WEIGHTS_DIR, name)
            if os.path.isfile(path):
                items.append({
                    "name": name,
                    "size": os.path.getsize(path),
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))),
                    "current": model_info.get("weights") == name,
                })
    return items


@app.route('/api/weights', methods=['GET'])
def get_weights():
    return jsonify({
        "weights": list_available_weights(),
        "current": model_info.get("weights"),
    })


@app.route('/api/weights/switch', methods=['POST'])
def switch_weights():
    global model_info
    data = request.get_json(silent=True) or {}
    name = str(data.get("weights", "")).strip()
    if not name:
        return jsonify({"error": "缺少权重文件名"}), 400

    weights_root = os.path.abspath(WEIGHTS_DIR)
    target = os.path.abspath(os.path.join(WEIGHTS_DIR, name))
    if not (target.startswith(weights_root + os.sep) and target.endswith(".pt") and os.path.isfile(target)):
        return jsonify({"error": f"非法的权重文件: {name}"}), 400

    with state_lock:
        if task_state["running"]:
            return jsonify({"error": "检测任务正在进行中，请先停止后再切换权重"}), 409

    old_weights = detector.WEIGHTS
    old_device = detector.device
    try:
        with model_lock:
            detector.WEIGHTS = target
            _free_model()
            detector.load_model()
            model_info = detector.get_model_info()
    except Exception as e:
        logger.exception("切换权重失败: %s", name)
        # 切换失败时回滚到原权重，避免检测器处于无模型状态
        try:
            detector.WEIGHTS = old_weights
            detector.device = old_device
            _free_model()
            detector.load_model()
            model_info = detector.get_model_info()
        except Exception:
            logger.exception("恢复原权重失败: %s", old_weights)
        return jsonify({"error": f"切换权重失败: {e}"}), 500

    return jsonify({"status": "ok", "info": model_info})


@app.route('/api/devices', methods=['GET'])
def get_devices():
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free = total = None
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:
                pass
            devices.append({
                "id": str(i),
                "name": torch.cuda.get_device_name(i),
                "free": free,
                "total": total,
            })
    devices.append({"id": "cpu", "name": "CPU", "free": None, "total": None})
    return jsonify({"devices": devices, "current": str(detector.device) if detector else None})


@app.route('/api/device/switch', methods=['POST'])
def switch_device():
    global model_info
    data = request.get_json(silent=True) or {}
    raw = str(data.get("device", "")).strip().lower()
    if not raw:
        return jsonify({"error": "缺少设备编号"}), 400

    if raw == "cpu":
        device = "cpu"
    else:
        try:
            idx = int(raw)
        except ValueError:
            return jsonify({"error": "无效的设备编号"}), 400
        if not torch.cuda.is_available():
            return jsonify({"error": "当前环境无可用 CUDA GPU"}), 400
        if idx < 0 or idx >= torch.cuda.device_count():
            return jsonify({"error": f"GPU {idx} 不存在（可用 0-{torch.cuda.device_count()-1}）"}), 400
        device = str(idx)

    with state_lock:
        if task_state["running"]:
            return jsonify({"error": "检测任务正在进行中，请先停止后再切换设备"}), 409

    # 设备切换只影响后续推理所用的设备，无需重新加载权重
    with model_lock:
        detector.device = device
        model_info = detector.get_model_info()

    return jsonify({"status": "ok", "info": model_info})


@app.route('/api/dashboard/start', methods=['POST'])
def start_task():
    global task_state
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400
    try:
        conf = float(request.form.get('conf', DEFAULT_CONF))
    except (TypeError, ValueError):
        conf = DEFAULT_CONF
    conf = max(0.05, min(0.95, conf))

    filepaths = []
    for f in files:
        fpath = os.path.join(UPLOAD_DIR, uuid.uuid4().hex + "_" + f.filename.replace("/", "_"))
        f.save(fpath)
        filepaths.append(fpath)

    with state_lock:
        if task_state["running"]:
            return jsonify({"error": "检测任务正在进行中，请先停止或等待完成"}), 409
        task_state.update({
            "running": True,
            "stop_requested": False,
            "files": filepaths,
            "current_idx": 0,
            "stats": {
                "success": 0, "fail": 0, "total": len(filepaths),
                "total_cells": 0, "total_conf": 0.0, "avg_conf": 0.0,
                "total_time": 0.0, "avg_time": 0.0,
                "rbc": 0, "wbc": 0,
            },
            "latest_result": {},
            "conf": conf,
            "logs": [f"[{time.strftime('%H:%M:%S')}] Start processing {len(filepaths)} images"],
        })
        threading.Thread(target=processing_loop, daemon=True).start()
    return jsonify({"status": "started", "total": len(filepaths)})


@app.route('/api/dashboard/stop', methods=['POST'])
def stop_task():
    with state_lock:
        task_state["stop_requested"] = True
        task_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] Stop requested")
    return jsonify({"status": "stopping"})


@app.route('/api/dashboard/status')
def get_status():
    with state_lock:
        stats = task_state["stats"]
        return jsonify({
            "running": task_state["running"],
            "conf": task_state.get("conf", DEFAULT_CONF),
            "progress": {"current": task_state["current_idx"], "total": stats["total"]},
            "stats": {
                "success": stats["success"],
                "fail": stats["fail"],
                "total": stats["total"],
                "total_cells": stats["total_cells"],
                "avg_conf": round(stats["avg_conf"], 4),
                "avg_time": round(stats["avg_time"] * 1000, 1),
                "rbc": stats["rbc"],
                "wbc": stats["wbc"],
            },
            "latest_result": task_state["latest_result"] if task_state["latest_result"] else None,
            "logs": task_state["logs"] if task_state["logs"] else [],
        })


@app.route('/api/result/image/<filename>')
def get_result_image(filename):
    safe_path = os.path.abspath(os.path.join(CACHE_DIR, filename))
    if safe_path.startswith(os.path.abspath(CACHE_DIR)) and os.path.isfile(safe_path):
        return send_file(safe_path)
    return jsonify({"error": "File not found"}), 404


@app.route('/api/history', methods=['GET'])
def get_history():
    keyword = request.args.get('keyword', '').lower().strip()
    with cache_lock:
        if not keyword:
            return jsonify(history_cache)

        filtered = []
        for item in history_cache:
            fname = str(item.get('filename', '')).lower()
            report = str(item.get('report', '')).lower()
            time_str = str(item.get('timestamp', '')).lower()
            total_str = str(item.get('total', ''))

            if (keyword in fname) or (keyword in report) or (keyword in time_str) or (keyword in total_str):
                filtered.append(item)

        return jsonify(filtered)


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    global history_cache
    with cache_lock:
        history_cache.clear()
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
    return jsonify({"status": "cleared", "message": "Detection history cleared."})


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )

    global detector, model_info, DEFAULT_CONF
    DEFAULT_CONF = args.conf

    logger.info("初始化 CF-YOLOv8 检测器...")
    detector = CellDetector(weights=args.weights, device=args.device, conf=args.conf)
    detector.load_model()
    model_info = detector.get_model_info()
    logger.info("模型加载完成: %s", model_info)

    logger.info("🩸 CF-YOLOv8 血细胞智能检测系统")
    logger.info("访问: http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
