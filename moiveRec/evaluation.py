"""
评估模块
负责计算各种评估指标 (RMSE, NDCG@K)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def calculate_rmse(y_true, y_pred):
    """
    计算均方根误差 (RMSE)
    
    Args:
        y_true: 真实评分
        y_pred: 预测评分
    
    Returns:
        float: RMSE值
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_ndcg_at_k(y_true, y_pred, user_ids, k=10):
    """
    计算NDCG@K
    对每个用户,根据预测分数排序,计算NDCG
    
    Args:
        y_true: 真实评分
        y_pred: 预测评分
        user_ids: 用户ID数组
        k: Top-K值 (默认10)
    
    Returns:
        float: 平均NDCG@K
    """
    df = pd.DataFrame({
        'user': user_ids,
        'true_rating': y_true,
        'pred_rating': y_pred
    })
    
    # 统计每个用户的评分数量分布
    user_rating_counts = {}
    for user in df['user'].unique():
        count = len(df[df['user'] == user])
        user_rating_counts[user] = count
    
    # 分组统计
    count_distribution = {i: 0 for i in range(1, 11)}
    count_distribution['10+'] = 0
    
    for count in user_rating_counts.values():
        if count >= 10:
            count_distribution['10+'] += 1
        else:
            count_distribution[count] += 1
    
    # 打印统计信息
    # print(f"用户评分数量分布:")
    # for i in range(1, 11):
    #     print(f"  评分数={i}: {count_distribution[i]} 个用户")
    # print(f"  评分数≥10: {count_distribution['10+']} 个用户")
    # print(f"  总用户数: {len(user_rating_counts)}")
    
    ndcg_scores = []
    
    for user in df['user'].unique():
        user_data = df[df['user'] == user].copy()
        
        # 如果用户的记录数少于2,跳过
        if len(user_data) < 2:
            continue
        
        # 按预测分数降序排序
        user_data = user_data.sort_values('pred_rating', ascending=False)
        
        # 取top-k 自动处理用户评分数量不足
        top_k = min(k, len(user_data))
        user_data_topk = user_data.head(top_k)
        
        # 计算DCG
        true_ratings = user_data_topk['true_rating'].values
        dcg = 0.0
        for i, rating in enumerate(true_ratings):
            dcg += (2**rating - 1) / np.log2(i + 2)  # i+2 because index starts at 0
        
        # 计算IDCG (理想情况下的DCG)
        ideal_ratings = sorted(user_data['true_rating'].values, reverse=True)[:top_k]
        idcg = 0.0
        for i, rating in enumerate(ideal_ratings):
            idcg += (2**rating - 1) / np.log2(i + 2)
        
        # 计算NDCG
        if idcg > 0:
            ndcg = dcg / idcg
            ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def evaluate_model(y_val, y_val_pred, val_user_ids, y_test, y_test_pred, test_user_ids):
    """
    全面评估模型性能
    
    Args:
        y_val: 验证集真实评分
        y_val_pred: 验证集预测评分
        val_user_ids: 验证集用户ID
        y_test: 测试集真实评分
        y_test_pred: 测试集预测评分
        test_user_ids: 测试集用户ID
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    print("\n[5/5] 评估模型...")
    
    # 计算RMSE
    rmse_val = calculate_rmse(y_val, y_val_pred)
    rmse_test = calculate_rmse(y_test, y_test_pred)
    
    print(f"【RMSE 指标】")
    print(f"  Validation RMSE: {rmse_val:.4f}")
    print(f"  Test RMSE:       {rmse_test:.4f}")
    
    # 计算NDCG@10
    ndcg_val = calculate_ndcg_at_k(y_val, y_val_pred, val_user_ids, k=10)
    ndcg_test = calculate_ndcg_at_k(y_test, y_test_pred, test_user_ids, k=10)
    
    print(f"【NDCG@10 指标】")
    print(f"  Validation NDCG@10: {ndcg_val:.4f}")
    print(f"  Test NDCG@10:       {ndcg_test:.4f}")
    
    return {
        'rmse_val': rmse_val,
        'rmse_test': rmse_test,
        'ndcg_val': ndcg_val,
        'ndcg_test': ndcg_test
    }
