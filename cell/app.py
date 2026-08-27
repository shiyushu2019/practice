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
import tempfile
import threading
import time
import uuid

from flask import Flask, render_template, request, jsonify, send_file

from core.cell_detector import CellDetector

HERE = os.path.dirname(os.path.abspath(__file__))

# 默认配置（可被命令行参数或环境变量覆盖）
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "cf.pt")
DEFAULT_CONF = 0.25
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000

logger = logging.getLogger("cell.dashboard")


def parse_args():
    parser = argparse.ArgumentParser(description="CF-YOLOv8 血细胞智能检测系统")
    parser.add_argument("--host", default=os.environ.get("CELL_HOST", DEFAULT_HOST),
                        help=f"监听地址（默认 {DEFAULT_HOST}，可用环境变量 CELL_HOST 覆盖）")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CELL_PORT", DEFAULT_PORT)),
                        help=f"监听端口（默认 {DEFAULT_PORT}，可用环境变量 CELL_PORT 覆盖）")
    parser.add_argument("--device", default=os.environ.get("CELL_DEVICE"),
                        help="推理设备（如 0-7 或 cpu），默认自动选择空闲显存最多的 GPU")
    parser.add_argument("--weights", default=os.environ.get("CELL_WEIGHTS", DEFAULT_WEIGHTS),
                        help=f"模型权重路径（默认 {DEFAULT_WEIGHTS}）")
    parser.add_argument("--conf", type=float, default=float(os.environ.get("CELL_CONF", DEFAULT_CONF)),
                        help=f"默认置信度阈值（默认 {DEFAULT_CONF}）")
    parser.add_argument("--debug", action="store_true", help="开启 Flask 调试模式")
    return parser.parse_args()


app = Flask(
    __name__,
    template_folder=os.path.join(HERE, "templates"),
    static_folder=os.path.join(HERE, "static"),
)
app.secret_key = os.environ.get("CELL_SECRET_KEY", "cell-dashboard-secret-key-2026")

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "cell_uploads")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "cell_result_cache")
os.makedirs(UPLOAD_DIR, exist_ok=True)
if os.path.exists(CACHE_DIR):
    try:
        shutil.rmtree(CACHE_DIR)
    except Exception as e:
        print(f"清理沙箱失败: {e}")
os.makedirs(CACHE_DIR, exist_ok=True)

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
        "rbc": 0, "wbc": 0, "platelets": 0,
    },
    "latest_result": {},
    "logs": [],
}
state_lock = threading.Lock()


def process_image(filepath, conf=0.25):
    """对单张图像执行检测并写入历史缓存。"""
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
                    task_state["stats"]["total_conf"] += result["avg_conf"]
                    task_state["stats"]["avg_conf"] = task_state["stats"]["total_conf"] / task_state["stats"]["success"]
                    task_state["stats"]["rbc"] += result["counts"]["RBC"]
                    task_state["stats"]["wbc"] += result["counts"]["WBC"]
                    task_state["stats"]["platelets"] += result["counts"]["Platelets"]
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
                task_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] Error {filename}: {str(e)}")
                task_state["current_idx"] += 1

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
                "rbc": 0, "wbc": 0, "platelets": 0,
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
                "platelets": stats["platelets"],
            },
            "latest_result": task_state["latest_result"] if task_state["latest_result"] else None,
            "logs": task_state["logs"] if task_state["logs"] else [],
        })


@app.route('/api/result/image/<filename>')
def get_result_image(filename):
    safe_path = os.path.abspath(os.path.join(CACHE_DIR, filename))
    if safe_path.startswith(os.path.abspath(CACHE_DIR)) and os.path.exists(safe_path):
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

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
