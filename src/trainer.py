#模型训练
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau




def train_model(model, train_loader, val_loader, num_classes, epochs=50, lr=0.001):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) #模型迁移至指定设备，模型将在GPU或CPU上运行。返回迁移后的模型
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()#损失函数，计算模型预测与真实标签之间的损失
    optimizer = optim.Adam(model.parameters(), lr=lr)#优化器，使用 Adam 优化算法，训练初期收敛快，model.parameters()参数为需要优化的模型参数
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)#学习调度器， 当指标（如损失或准确率）停滞时，自动降低学习率
    #min代表指标越小越好，patience验证集损失连续 5 轮未下降，学习率将降低。verbose=True参数不支持，pytorch版本低于1.10
    
    
    # 记录最佳验证准确率
    best_val_acc = 0.0#记录验证集上的最高准确率，记录最佳状态可防止使用性能退化的模型
    best_model_weights = None#保存达到该准确率时的模型参数。
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
             # 数据到设备
            features, labels = features.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad() # 清空梯度，防止梯度累加，确保每次迭代的梯度仅来自当前批次
            outputs = model(features)  # 模型预测
            loss = criterion(outputs, labels)# 计算损失
            
            # 反向传播和优化
            loss.backward()# 计算梯度，从损失函数反向传播，计算每个可训练参数的梯度
            optimizer.step() # 更新参数，根据优化器算法（Adam）更新模型参数
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)   #max(1)返回每一行的最大值及索引
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 验证模式
        model.eval()#关闭 Dropout 和 BatchNorm 的训练模式。Dropout 在验证时不随机丢弃神经元。BatchNorm 使用训练集的统计量（均值、方差）
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():#停止梯度计算，节省内存并加速计算。验证阶段无需更新参数，无需梯度信息
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 计算平均损失和准确率
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        # 基于验证损失调整学习率，每轮验证后，若验证损失未下降，学习率降低，步长变小，精细化的调整参数
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict()
            print(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%")
    
    # 加载最佳模型权重
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    
    return model, best_val_acc

def save_model(model, path='models/traffic_model.pth'):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存至 {path}")