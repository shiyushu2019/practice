"""血细胞检测 Web UI (Flask)。

用法:
    python app.py
    然后浏览器访问 http://127.0.0.1:5000
"""
import os
import uuid
import base64

from flask import Flask, request, jsonify

from ultralytics import YOLO

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights", "cf.pt")

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

print("🩸 加载模型:", WEIGHTS)
model = YOLO(WEIGHTS)
print("✅ 加载成功！")

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>血细胞检测</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 50px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .upload-btn { display: inline-block; padding: 15px 40px; background: #667eea; color: white; border-radius: 10px; cursor: pointer; margin: 20px 0; border: none; font-size: 16px; }
        .upload-btn:hover { background: #5a67d8; }
        #file-input { display: none; }
        .result { margin-top: 30px; display: none; }
        .result img { max-width: 100%; border-radius: 10px; }
        .stats { display: flex; justify-content: center; gap: 30px; margin: 20px 0; }
        .stat { padding: 10px 20px; background: #f0f0f0; border-radius: 10px; }
        .stat .num { font-size: 24px; font-weight: bold; }
        .stat .label { font-size: 14px; color: #888; }
        .stat.rbc .num { color: #e74c3c; }
        .stat.wbc .num { color: #3498db; }
        .stat.platelets .num { color: #2ecc71; }
        .loading { display: none; color: #888; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩸 CF-YOLOv8 血细胞检测</h1>
        <p>上传血细胞图片，自动识别红细胞、白细胞、血小板</p>
        <button class="upload-btn" onclick="document.getElementById('file-input').click()">📤 选择图片</button>
        <input type="file" id="file-input" accept="image/*">
        <div class="loading" id="loading">⏳ 检测中...</div>
        <div class="result" id="result">
            <img id="result-img" src="">
            <div class="stats">
                <div class="stat rbc"><div class="num" id="rbc-num">0</div><div class="label">RBC 红细胞</div></div>
                <div class="stat wbc"><div class="num" id="wbc-num">0</div><div class="label">WBC 白细胞</div></div>
                <div class="stat platelets"><div class="num" id="plt-num">0</div><div class="label">Platelets 血小板</div></div>
            </div>
            <p id="detail"></p>
            <button onclick="location.reload()">🔄 继续检测</button>
        </div>
    </div>
    <script>
        document.getElementById('file-input').addEventListener('change', function(e) {
            var file = this.files[0];
            if (!file) return;
            var formData = new FormData();
            formData.append('file', file);
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            fetch('/detect', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(d => {
                    document.getElementById('loading').style.display = 'none';
                    if (d.error) { alert(d.error); return; }
                    document.getElementById('result-img').src = 'data:image/jpeg;base64,' + d.image;
                    document.getElementById('rbc-num').textContent = d.counts.RBC;
                    document.getElementById('wbc-num').textContent = d.counts.WBC;
                    document.getElementById('plt-num').textContent = d.counts.Platelets;
                    document.getElementById('detail').textContent = '共检测到 ' + d.total + ' 个细胞';
                    document.getElementById('result').style.display = 'block';
                })
                .catch(e => { alert('检测失败: ' + e.message); document.getElementById('loading').style.display = 'none'; });
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return HTML


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    if not file:
        return jsonify({'error': '没有文件'})

    filename = str(uuid.uuid4()) + '.jpg'
    filepath = os.path.join('uploads', filename)
    file.save(filepath)

    results = model(filepath, conf=0.25)

    counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
    for box in results[0].boxes:
        cls = int(box.cls)
        if cls == 0:
            counts['RBC'] += 1
        elif cls == 1:
            counts['WBC'] += 1
        elif cls == 2:
            counts['Platelets'] += 1

    result_path = os.path.join('static/results', filename)
    results[0].save(filename=result_path)

    with open(result_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode()

    return jsonify({
        'image': img_data,
        'counts': counts,
        'total': len(results[0].boxes),
    })


if __name__ == '__main__':
    print("=" * 40)
    print("🩸 CF-YOLOv8 血细胞检测系统")
    print("🌐 访问: http://127.0.0.1:5000")
    print("=" * 40)
    app.run(debug=False, host='127.0.0.1', port=5000)
