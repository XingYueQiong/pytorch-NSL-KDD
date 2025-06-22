#工具函数定义
import os
import yaml
import joblib

def load_config(config_path='config/model_config.yaml'):    #用于加载YAML格式的配置文件，config_path参数指定文件路径，不传入参数也可，会默认读取该文件
    """加载配置文件"""
    with open(config_path, 'r',encoding='utf-8') as f:   #以读取方式打开文件，将文件对象赋值给变量f。操作完文件后自动关闭，无需f.close()
        config = yaml.safe_load(f)  #使用safe_load方法解析文件内容，解析后转化为字典赋值给config变量。依赖yaml库。
    return config

def ensure_dirs(dirs):  #传入一个目录路径的列表
    """确保目录存在"""
    for dir_path in dirs:   #循环遍历dirs列表下的每个路径
        if not os.path.exists(dir_path):    #使用os.path.exists()函数检查目录是否存在，不存在返回false
            os.makedirs(dir_path)   #不存在则创建目录

            