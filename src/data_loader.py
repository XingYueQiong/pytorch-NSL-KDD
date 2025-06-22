#数据加载与预处理
import os
from sklearn.model_selection import train_test_split
import torch
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler



class NSLKDDDataset(Dataset):#NSLKDDDataset继承自 PyTorch 的torch.utils.data.Dataset基类。封装特征和标签，提供统一的数据访问接口
   
    """NSL-KDD数据集加载器"""
    def __init__(self, features, labels):#将 NumPy 数组转换为 PyTorch 张量
        self.features = torch.FloatTensor(features)#特征转为 32 位浮点数。

# 将 Pandas Series 转换为 NumPy 数组
        if isinstance(labels, pd.Series):
            labels = labels.values

        self.labels = torch.LongTensor(labels)#标签转为 64 位整数（多分类要求）. torch.LongTensor() 不直接支持 Series,所以在上一步添加转换
        
    def __len__(self):#返回数据集大小（样本数）
        return len(self.labels)
    
    def __getitem__(self, idx):#支持通过索引访问样本。返回元组(features, label)，适配 PyTorch 模型输入
        return self.features[idx], self.labels[idx]

def load_data(data_dir, combine_and_split=True, test_size=0.2, random_state=42):
    """从data_dir目录加载Train训练集和Test测试集"""
    # 定义数据集的列名
    columns = [
        'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
      'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
       'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
       'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
       'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
       'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
       'dst_host_serror_rate', 'dst_host_srv_serror_rate',
       'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'duration'
    ]

    # 构建文件路径，加载训练集和测试集
    train_file = os.path.join(data_dir, 'KDDTrain+.txt')
    test_file = os.path.join(data_dir, 'KDDTest+.txt')
    # 读取数据，header=None表示文件无表头，names指定列名
    train_df = pd.read_csv(train_file, header=None, names=columns)
    test_df = pd.read_csv(test_file, header=None, names=columns)

    if combine_and_split:
        # 合并数据集
        combined_data = pd.concat([train_df, test_df], axis=0)
        print(f"合并后数据集大小: {len(combined_data)} 条记录")

        # 强制转换duration为数值类型
        combined_data['duration'] = pd.to_numeric(combined_data['duration'], errors='coerce')

        # 清洗label列
        combined_data['label'] = combined_data['label'].str.split(',').str[0]

        # 重新排列列顺序，将duration移到第一列
        columns = ['duration'] + columns[:-1]
        combined_data = combined_data[columns]

        # 手动划分数据
        X = combined_data.drop('label', axis=1)
        y = combined_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

    else:
        # 强制转换duration为数值类型
        train_df['duration'] = pd.to_numeric(train_df['duration'], errors='coerce')
        test_df['duration'] = pd.to_numeric(test_df['duration'], errors='coerce')

        # 清洗label列
        train_df['label'] = train_df['label'].str.split(',').str[0]
        test_df['label'] = test_df['label'].str.split(',').str[0]

        # 重新排列列顺序，将duration移到第一列
        columns = ['duration'] + columns[:-1]
        train_df = train_df[columns]
        test_df = test_df[columns]

    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")

    return train_df, test_df

def create_preprocessor():
    """创建数据预处理流水线"""
    # 定义特征类型
    categorical_features = ['protocol_type', 'service', 'flag']#分类特征，需转换为数值,三分类
    numerical_features = [ #数值特征，需标准化
        'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
        'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    
    # 创建预处理流水线，确保数据格式的一致性和模型输入的有效性
    preprocessor = Pipeline([ #Pipeline（）将多个数据处理步骤串联为单一对象，[('步骤名', 转换器), ...]。
        ('column_transformer', ColumnTransformer([# ColumnTransformer()对不同列应用不同的转换器。下面的参数remainder='drop'：丢弃未指定的列
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),# OneHotEncoder()将分类特征转换为二进制向量。protocol_type（TCP/UDP/ICMP）→ 3 个二进制列，drop='first'：删除第一个类别，handle_unknown='ignore'：忽略未知类别。默认输出稀疏矩阵
            ('num', MinMaxScaler(), numerical_features)#数值特征处理，保留数据的比例，X_scaled = (X - X_min) / (X_max - X_min)
        ], remainder='drop')),
    ])
    
    return preprocessor
#实现多分类任务，保留原始标签类别
def prepare_data(train_df, test_df, save_preprocessor=True):
    """准备训练和测试数据"""
    # 提取特征和标签 (多分类标签)。X为特征，Y为标签
    X_train = train_df.drop('label', axis=1)#从train_fd中删除label列，axis=1删除列，axis=0删除行
    y_train = train_df['label']#从train_df中单独提取label列
    
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']
    
    # 标签编码，输出形式为整数索引如0、1。
    # 获取唯一标签类别并排序，sorted()保证映射顺序确定性，用于双向转换
    all_labels = sorted(y_train.unique())#提取训练集中所有唯一的标签值，按字母顺序排序
    label_to_idx = {label: i for i, label in enumerate(all_labels)}#构建字典，将每个标签映射到一个唯一的整数索引
    
    # 转换标签为数字
   # y_train = y_train.map(label_to_idx).values
   # y_test = y_test.map(label_to_idx).values

    y_train = y_train.map(label_to_idx).fillna(0).astype(int)  # 处理缺失标签
    y_test = y_test.map(label_to_idx).fillna(0).astype(int)

    # 创建并拟合预处理流水线
    preprocessor = create_preprocessor()#函数返回一个pipeline对象，包含数值特征的标准化和分类特征的编码
    X_train_processed = preprocessor.fit_transform(X_train)#在训练集上拟合预处理参数（如均值、标准差），并应用转换
    X_test_processed = preprocessor.transform(X_test)#使用训练集的参数转换测试集（避免数据泄露）
    
    # 保存预处理流水线，确保新数据的预处理方式与训练数据一致
    if save_preprocessor:
        joblib.dump(preprocessor, 'models/preprocessor.pkl')#使用joblib库将预处理流水线化为二进制文件
        joblib.dump(label_to_idx, 'models/label_mapping.pkl')
    
    # 创建PyTorch数据集和数据加载器
    train_dataset = NSLKDDDataset(X_train_processed, y_train)
    test_dataset = NSLKDDDataset(X_test_processed, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)#每次加载 64 条样本.shuffle=True：训练集随机打乱，提高泛化能力
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)#shuffle=False：测试集保持顺序，便于结果比对。
    
    return (train_loader, test_loader, X_train_processed, y_train, 
            X_test_processed, y_test, preprocessor, label_to_idx)
