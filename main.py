import os
import torch
#导入src目录下的文件中的函数
from src.data_loader import load_data, prepare_data 
from src.model import get_model
from src.trainer import train_model, save_model
from src.evaluator import evaluate_model
from src.utils import load_config, ensure_dirs
"""X为特征，Y为标签"""

def main():
    # 确保目录存在，不存在的话直接就在函数里创建
    ensure_dirs(['models', 'reports'])
    
    # 加载配置
    config = load_config()
    epochs = config.get('epochs', 50)#从配置字典中获取训练轮数（epochs），若不存在则使用默认值 50。
    lr = config.get('learning_rate', 0.001)#同上
    
    # 加载数据
    print("加载数据...")
    #start
    train_df, test_df = load_data('data', combine_and_split=True, test_size=0.2, random_state=42)# 合并并重新划分
   # train_df, test_df = load_data('data')#从data路径读取数据集并拆分为训练集和测试集，函数返回两个元素的列表
    """预计输出：训练集形状: (125973, 42)
                测试集形状: (22544, 42)。label列单独被提取，"""

    # 准备数据，多分类处理，返回8个值。将原始数据转换为模型可用的格式。
    print("数据预处理...")
    (train_loader, test_loader, X_train, y_train, #pytorch数据加载器、特征矩阵（输入）、标签向量（输出)、预处理流水线、标签到索引的映射字典
     X_test, y_test, preprocessor, label_to_idx) = prepare_data(train_df, test_df)
    
    # 获取模型，准备模型训练所需的参数并获取模型实例
    input_dim = X_train.shape[1]#获取特征维度（输入层大小），获取特征矩阵的列数。类别特征编码，比如service类别有70种特征，被独热编码为70列。
    num_classes = len(set(y_train))#获取类别数量（输出层大小）,计算唯一标签的数量,set(y_train)：将标签转换为集合（去重）。23个不同类别的标签。
    print(f"输入维度: {input_dim}, 类别数: {num_classes}")
    model = get_model(input_dim, num_classes)# get_model() 函数创建模型
    
    # 训练模型
    print("训练模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#设备配置，若GPU可用则用GPU，否则使用cpu   torch.device()：创建一个表示设备的对象
    model, best_acc = train_model(                  #函数返回值model:训练好的pytorch模型      best_acc:浮点数，验证准确率
        model, train_loader, test_loader, num_classes, #调用 train_model 函数，传入模型、数据加载器和训练参数
        epochs=epochs, lr=lr#num_classes:分类任务的类别数   epochs：训练轮数    lr：学习率，控制模型参数更新的幅度
    )
    
    # 保存模型
    save_model(model)
    
    # 评估模型
    print("评估模型...")
    metrics = evaluate_model(
        model, test_loader, y_test, label_to_idx, device
    )
    
    # 检查准确率是否达到98%
    if metrics['accuracy'] >= 0.98:
        print(f"成功达到目标准确率: {metrics['accuracy']:.4f}")
    else:
        print(f"当前准确率: {metrics['accuracy']:.4f}, 未达到98%目标")
        print("建议调整模型结构或超参数")
    
    # 保存评估结果
    with open('reports/classification_report.txt', 'w') as f:
        f.write("===== 评估指标 =====\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
        f.write(f"宏平均精确率 (Macro Precision): {metrics['macro_precision']:.4f}\n")
        f.write(f"宏平均召回率 (Macro Recall): {metrics['macro_recall']:.4f}\n")
        f.write(f"宏平均F1分数 (Macro F1): {metrics['macro_f1']:.4f}\n")
        f.write(f"宏平均ROC-AUC: {metrics['roc_auc_macro']:.4f}\n")
        
        f.write("\n===== 各类别评估指标 =====\n")
        for label, metrics_dict in metrics['class_metrics'].items():
            f.write(f"{label}:\n")
            f.write(f"  精确率: {metrics_dict['precision']:.4f}\n")
            f.write(f"  召回率: {metrics_dict['recall']:.4f}\n")
            f.write(f"  F1分数: {metrics_dict['f1']:.4f}\n")
            f.write(f"  ROC-AUC: {metrics_dict['roc_auc']:.4f}\n")

if __name__ == "__main__":
    main()
    