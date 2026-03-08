"""
模型训练模块
负责训练逻辑回归模型
"""

import os
import pickle
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train_class, verbose=True):
    """
    训练多分类逻辑回归模型
    
    Args:
        X_train: 训练特征
        y_train_class: 训练标签 (0-4)
        verbose: 是否显示训练详情
    
    Returns:
        LogisticRegression: 训练好的模型
    """
    print("\n[4/5] 训练逻辑回归模型...")
    print("  注: 将评分预测转换为分类问题 (Rating: 1-5 -> Class: 0-4)")
    
    # 训练多分类逻辑回归
    lr_model = LogisticRegression(
        multi_class='multinomial',  # 多分类策略: multinomial使用softmax函数,适合多分类问题
        solver='lbfgs',             # 优化算法: lbfgs是拟牛顿法,适合小到中等数据集
        max_iter=1000,              # 最大迭代次数: 优化器的最大迭代轮数
        random_state=42,            # 随机种子: 确保结果可复现
        n_jobs=-1,                  # 并行数: -1表示使用所有CPU核心
        verbose=1 if verbose else 0 # 详细程度: 1输出训练过程信息,0不输出
    )
    
    print("  开始训练...")
    lr_model.fit(X_train, y_train_class)
    print("  模型训练完成")
    
    if verbose:
        print(f"  实际迭代次数: {lr_model.n_iter_[0] if hasattr(lr_model, 'n_iter_') else 'N/A'}")
        print(f"  类别数量: {len(lr_model.classes_)}")
    
    return lr_model


def predict(model, X):
    """
    使用模型进行预测
    
    Args:
        model: 训练好的模型
        X: 特征矩阵
    
    Returns:
        array: 预测概率 (n_samples, 5)
    """
    return model.predict_proba(X)


def save_model(model, filepath='models/lr_model.pkl'):
    """
    保存训练好的模型
    
    Args:
        model: 训练好的模型
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"  模型已保存至: {filepath}")


def load_model(filepath='models/lr_model.pkl'):
    """
    加载训练好的模型
    
    Args:
        filepath: 模型文件路径
    
    Returns:
        model: 加载的模型,如果文件不存在则返回None
    """
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"  已加载保存的模型: {filepath}")
        return model
    return None
