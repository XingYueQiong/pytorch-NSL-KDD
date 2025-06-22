#pytorch模型定义
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficClassifier(nn.Module):
    """网络流量分类模型"""
    def __init__(self, input_dim, num_classes):
        super(TrafficClassifier, self).__init__()
        # 定义神经网络层
        self.fc1 = nn.Linear(input_dim, 256)#四层全连接神经网络（fc1-fc4），输入维度input_dim→输出 256，完成输入特征到隐藏层的映射
        self.bn1 = nn.BatchNorm1d(256)       #3个批量归一化层（bn1-bn3），对每个隐藏层的输出进行归一化，加速训练并提高稳定性
        self.fc2 = nn.Linear(256, 128)      #256→128，进一步提取特征
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)       #128→64，降低特征维度
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)#64→num_classes，输出分类结果（23 个类别）
        self.dropout = nn.Dropout(0.2)  # Dropout 层（防止过拟合）0.3 可调整为 0.2 或 0.4，视过拟合情况而定
        
    def forward(self, x):
        """前向传播"""
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

def get_model(input_dim, num_classes):
    """获取模型实例"""
    model = TrafficClassifier(input_dim, num_classes)
    return model

