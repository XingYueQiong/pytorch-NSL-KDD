#模型评估
import torch
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, test_loader, y_true, label_to_idx, device):
    """评估模型性能"""
    model.eval()#设置模型为评估模式
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad(): # 关闭梯度计算
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            #将数据从 GPU 转回 CPU 并转换为 NumPy 数组
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
      # 获取测试集中实际出现的类别
    unique_classes = np.unique(all_labels)
    num_classes = len(label_to_idx)
    # 计算四大评估指标
    accuracy = accuracy_score(all_labels, all_preds)#准确率
    
    # 处理多分类的精确率、召回率和F1分数
    precision = precision_score(all_labels, all_preds, average=None,zero_division=1)#zero_division=0（默认）：分母为 0 时设为 0，发出警告     zero_division=1：分母为 0 时设为 1，不发出警告
    recall = recall_score(all_labels, all_preds, average=None,zero_division=1)
    f1 = f1_score(all_labels, all_preds, average=None,zero_division=1)
    
    # 计算宏平均和微平均
    precision_macro = precision_score(all_labels, all_preds, average='macro',zero_division=1)#计算每个类别的指标后取算术平均，适合类别不平衡场景
    recall_macro = recall_score(all_labels, all_preds, average='macro',zero_division=1)
    f1_macro = f1_score(all_labels, all_preds, average='macro',zero_division=1)
    
    precision_micro = precision_score(all_labels, all_preds, average='micro',zero_division=1)#将所有类别样本合并计算指标，适合关注整体性能场景
    recall_micro = recall_score(all_labels, all_preds, average='micro',zero_division=1)
    f1_micro = f1_score(all_labels, all_preds, average='micro',zero_division=1)
    
    
    # 计算ROC-AUC (多分类情况)
    roc_auc = {}
    for class_idx in unique_classes:
        # 对每个类别计算One-vs-Rest的AUC
        y_onehot = np.zeros((len(all_labels), num_classes))
        y_onehot[np.arange(len(all_labels)), all_labels] = 1
        try:
            roc_auc[class_idx] = roc_auc_score(y_onehot[:, class_idx], all_probs[:, class_idx])
        except:
            roc_auc[class_idx] = 0.5  # 处理只有一类的情况
    
    roc_auc_macro = np.mean(list(roc_auc.values())) if roc_auc else 0.5
    
    # 打印评估指标
    print("\n===== 评估指标 =====")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"宏平均精确率 (Macro Precision): {precision_macro:.4f}")
    print(f"宏平均召回率 (Macro Recall): {recall_macro:.4f}")
    print(f"宏平均F1分数 (Macro F1): {f1_macro:.4f}")
    print(f"微平均精确率 (Micro Precision): {precision_micro:.4f}")
    print(f"微平均召回率 (Micro Recall): {recall_micro:.4f}")
    print(f"微平均F1分数 (Micro F1): {f1_micro:.4f}")
    print(f"宏平均ROC-AUC: {roc_auc_macro:.4f}")
    
    # 打印每个类别的指标
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    print("\n===== 各类别评估指标 =====")
    for class_idx in unique_classes:
        class_name = idx_to_label.get(class_idx, f"未知类别_{class_idx}")
        print(f"{class_name}:")
        # 找到该类别在评估指标数组中的位置
        metric_idx = np.where(unique_classes == class_idx)[0][0]
        print(f"  精确率: {precision[metric_idx]:.4f}")
        print(f"  召回率: {recall[metric_idx]:.4f}")
        print(f"  F1分数: {f1[metric_idx]:.4f}")
        print(f"  ROC-AUC: {roc_auc.get(class_idx, 0.5):.4f}")
    # 处理测试集中未出现的类别
    missing_classes = set(label_to_idx.values()) - set(unique_classes)
    if missing_classes:
        print("\n===== 测试集中未出现的类别 =====")
        for class_idx in missing_classes:
            class_name = idx_to_label.get(class_idx, f"未知类别_{class_idx}")
            print(f"{class_name}:")
            print(f"  精确率: 0.0000")
            print(f"  召回率: 0.0000")
            print(f"  F1分数: 0.0000")
            print(f"  ROC-AUC: 0.5000")
    # 绘制混淆矩阵（需要调整以处理实际类别）
    plot_confusion_matrix(all_labels, all_preds, idx_to_label, unique_classes)
    
    return {
        'accuracy': accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'micro_precision': precision_micro,
        'micro_recall': recall_micro,
        'micro_f1': f1_micro,
        'roc_auc_macro': roc_auc_macro,
        'class_metrics': {
            idx_to_label[class_idx]: {
                'precision': precision[np.where(unique_classes == class_idx)[0][0]] if class_idx in unique_classes else 0.0,
                'recall': recall[np.where(unique_classes == class_idx)[0][0]] if class_idx in unique_classes else 0.0,
                'f1': f1[np.where(unique_classes == class_idx)[0][0]] if class_idx in unique_classes else 0.0,
                'roc_auc': roc_auc.get(class_idx, 0.5)
            } for class_idx in label_to_idx.values()
        }
    }

def plot_confusion_matrix(y_true, y_pred, label_dict, unique_classes):
       # 自动查找系统中已安装的中文字体
    chinese_fonts = []
    for font in fm.fontManager.ttflist:
        # 检查字体是否支持中文（通过名称包含中文或支持CJK字符）
        if "SimHei" in font.name or "Microsoft YaHei" in font.name or any(ord(c) > 127 for c in font.name):
            chinese_fonts.append(font.name)
    
    if chinese_fonts:
        plt.rcParams["font.family"] = chinese_fonts
        print(f"使用字体: {chinese_fonts[0]}")
    else:
        plt.rcParams["font.family"] = ["sans-serif"]
        print("警告：未找到中文字体，使用默认字体")
    
    """绘制混淆矩阵（支持实际出现的类别）"""
    cm = confusion_matrix(y_true, y_pred)
    
    # 确保标签顺序与实际出现的类别一致
    labels = [label_dict[i] for i in sorted(unique_classes)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png')
    plt.close()