# -*- coding: utf-8 -*-
"""CF-YOLOv8 血细胞智能检测系统 (Flask Dashboard)。

参照 web_app_example 的架构与 UI 风格重写。

用法:
    python app.py
    浏览器访问 http://127.0.0.1:5000
"""
import os
import base64
import threading
import time
import shutil
import uuid
import tempfile

from flask import Flask, render_template, request, jsonify, send_file

from core.cell_detector import CellDetector

HERE = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(HERE, "templates"),
    static_folder=os.path.join(HERE, "static"),
)
app.secret_key = 'cell-dashboard-secret-key-2026'

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "cell_uploads")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "cell_privacy_cache")
os.makedirs(UPLOAD_DIR, exist_ok=True)
if os.path.exists(CACHE_DIR):
    try:
        shutil.rmtree(CACHE_DIR)
    except Exception as e:
        print(f"清理沙箱失败: {e}")
os.makedirs(CACHE_DIR, exist_ok=True)

history_cache = []
cache_lock = threading.Lock()

print("Initializing CF-YOLOv8 Detector...")
detector = CellDetector()
detector.load_model()
model_info = detector.get_model_info()
print("All detection models loaded:", model_info)

task_state = {
    "running": False,
    "stop_requested": False,
    "files": [],
    "current_idx": 0,
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


def process_image(filepath):
    """对单张图像执行检测并写入历史缓存。"""
    result = detector.detect({"image_path": filepath})
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
        "orig_url": f"/api/sandbox/image/{orig_name}",
        "result_url": f"/api/sandbox/image/{result_name}",
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
            task_state["stats"]["total"] = len(task_state["files"])

        start_proc_time = time.time()
        try:
            result = process_image(filepath)
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

    filepaths = []
    for f in files:
        fpath = os.path.join(UPLOAD_DIR, uuid.uuid4().hex + "_" + f.filename.replace("/", "_"))
        f.save(fpath)
        filepaths.append(fpath)

    with state_lock:
        if task_state["running"]:
            task_state["stop_requested"] = True
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


@app.route('/api/sandbox/image/<filename>')
def get_sandbox_image(filename):
    safe_path = os.path.abspath(os.path.join(CACHE_DIR, filename))
    if safe_path.startswith(os.path.abspath(CACHE_DIR)) and os.path.exists(safe_path):
        return send_file(safe_path)
    return jsonify({"error": "File burned"}), 404


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


@app.route('/api/history/burn', methods=['POST'])
def burn_history():
    global history_cache
    with cache_lock:
        history_cache.clear()
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
    return jsonify({"status": "burned", "message": "Memory and disk traces safely erased."})


if __name__ == '__main__':
    print("=" * 40)
    print("🩸 CF-YOLOv8 血细胞智能检测系统")
    print("🌐 访问: http://127.0.0.1:5000")
    print("=" * 40)
    app.run(host='0.0.0.0', port=5000, debug=False)
