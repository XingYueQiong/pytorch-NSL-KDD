#API服务
import torch
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from src.model import TrafficClassifier

app = Flask(__name__)

# 加载模型和预处理工具
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocessor = joblib.load('models/preprocessor.pkl')
label_mapping = joblib.load('models/label_mapping.pkl')
idx_to_label = {v: k for k, v in label_mapping.items()}

# 加载模型
input_dim = preprocessor.transform(pd.DataFrame({
    'duration': [0],
    'protocol_type': ['tcp'],
    'service': ['http'],
    'flag': ['SF'],
    'src_bytes': [0],
    'dst_bytes': [0],
    'land': [0],
    'wrong_fragment': [0],
    'urgent': [0],
    'hot': [0],
    'num_failed_logins': [0],
    'logged_in': [0],
    'num_compromised': [0],
    'root_shell': [0],
    'su_attempted': [0],
    'num_root': [0],
    'num_file_creations': [0],
    'num_shells': [0],
    'num_access_files': [0],
    'count': [0],
    'srv_count': [0],
    'serror_rate': [0.0],
    'srv_serror_rate': [0.0],
    'rerror_rate': [0.0],
    'srv_rerror_rate': [0.0],
    'same_srv_rate': [0.0],
    'diff_srv_rate': [0.0],
    'srv_diff_host_rate': [0.0],
    'dst_host_count': [0],
    'dst_host_srv_count': [0],
    'dst_host_same_srv_rate': [0.0],
    'dst_host_diff_srv_rate': [0.0],
    'dst_host_same_src_port_rate': [0.0],
    'dst_host_srv_diff_host_rate': [0.0],
    'dst_host_serror_rate': [0.0],
    'dst_host_srv_serror_rate': [0.0],
    'dst_host_rerror_rate': [0.0],
    'dst_host_srv_rerror_rate': [0.0]
})).shape[1]

num_classes = len(label_mapping)
model = TrafficClassifier(input_dim, num_classes)
model.load_state_dict(torch.load('models/traffic_model.pth', map_location=device))
model = model.to(device)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    """预测网络流量类型"""
    try:
        # 获取请求数据
        data = request.json
        
        # 转换为DataFrame
        input_df = pd.DataFrame([data])
        
        # 预处理
        processed_data = preprocessor.transform(input_df)
        processed_tensor = torch.FloatTensor(processed_data).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(processed_tensor)
            probs = torch.softmax(outputs, 1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
        
        # 返回结果
        result = {
            'prediction': int(pred_idx),
            'label': idx_to_label[pred_idx],
            'probabilities': {idx_to_label[i]: float(probs[i]) for i in range(num_classes)},
            'confidence': f"{float(probs[pred_idx]) * 100:.2f}%"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)