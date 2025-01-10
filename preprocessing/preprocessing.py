import os
import numpy as np
import pandas as pd

def generate_features():
    """生成特征数据"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义原始数据目录
    data_dir = os.path.join(current_dir, "data")
    
    # 如果数据目录不存在则创建
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    try:
        # 读取原始数据
        c1_data = pd.read_csv(os.path.join(data_dir, "c1.csv"))
        c4_data = pd.read_csv(os.path.join(data_dir, "c4.csv"))
        c6_data = pd.read_csv(os.path.join(data_dir, "c6.csv"))
        
        # 处理数据并生成特征
        # ... 特征处理代码 ...
        
        # 保存特征数据
        feature_files = [
            'c1_features_0.npy', 'c1_features_1.npy', 'c1_features_2.npy',
            'c1_features_3.npy', 'c1_features_4.npy', 'c1_features_5.npy',
            'c1_features_6.npy', 'c1_features_7.npy', 'c1_features_8.npy',
            'c1_features.npy',
            'c4_features_0.npy', 'c4_features_1.npy', 'c4_features_2.npy',
            'c4_features_3.npy', 'c4_features_4.npy', 'c4_features_5.npy',
            'c4_features_6.npy', 'c4_features_7.npy', 'c4_features_8.npy',
            'c4_features.npy',
            'c6_features_0.npy', 'c6_features_1.npy', 'c6_features_2.npy',
            'c6_features_3.npy', 'c6_features_4.npy', 'c6_features_5.npy',
            'c6_features_6.npy', 'c6_features_7.npy', 'c6_features_8.npy',
            'c6_features.npy'
        ]
        
        # 生成示例特征数据(这里需要替换为实际的特征生成逻辑)
        for feature_file in feature_files:
            # 生成随机数据作为示例
            data = np.random.rand(315, 7, 28).astype('float32')
            np.save(os.path.join(current_dir, feature_file), data)
            
        print("特征生成完成")
        
    except Exception as e:
        print(f"特征生成失败: {e}")
        raise

if __name__ == "__main__":
    generate_features()

