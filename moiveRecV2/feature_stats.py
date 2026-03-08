"""
特征统计模块
提供用户行为统计和电影热度统计功能
"""

import pandas as pd
import numpy as np


def calculate_user_stats(train_data):
    """
    计算用户行为统计特征
    
    Args:
        train_data: 训练集DataFrame
        
    Returns:
        DataFrame: 用户统计特征
            - user_rating_mean: 用户平均评分
            - user_rating_std: 用户评分标准差
            - user_rating_count: 用户评分总数
    """
    user_stats = train_data.groupby('UserID')['Rating'].agg([
        ('user_rating_mean', 'mean'),      # 用户平均评分
        ('user_rating_std', 'std'),        # 用户评分标准差
        ('user_rating_count', 'count')     # 用户评分总数
    ]).reset_index()
    
    # 填充标准差的NaN值(只有1次评分的用户)
    user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
    
    return user_stats


def calculate_movie_stats(train_data):
    """
    计算电影热度特征
    
    Args:
        train_data: 训练集DataFrame
        
    Returns:
        DataFrame: 电影统计特征
            - movie_rating_mean: 电影平均评分
            - movie_rating_std: 电影评分标准差
            - movie_rating_count: 电影评分总数(热度)
    """
    movie_stats = train_data.groupby('MovieID')['Rating'].agg([
        ('movie_rating_mean', 'mean'),     # 电影平均评分
        ('movie_rating_std', 'std'),       # 电影评分标准差
        ('movie_rating_count', 'count')    # 电影评分总数(热度)
    ]).reset_index()
    
    # 填充标准差的NaN值(只有1次评分的电影)
    movie_stats['movie_rating_std'] = movie_stats['movie_rating_std'].fillna(0)
    
    return movie_stats


def create_unk_stats(user_stats, movie_stats, unk_user_id, unk_movie_id):
    """
    为UNK用户和电影创建统计特征(使用全局平均值)
    
    Args:
        user_stats: 用户统计特征DataFrame
        movie_stats: 电影统计特征DataFrame
        unk_user_id: UNK用户ID
        unk_movie_id: UNK电影ID
        
    Returns:
        tuple: (user_stats_with_unk, movie_stats_with_unk)
    """
    # 为UNK用户创建统计特征(使用全局平均值)
    global_user_mean = user_stats['user_rating_mean'].mean()
    global_user_std = user_stats['user_rating_std'].mean()
    global_user_count = user_stats['user_rating_count'].mean()
    
    unk_user_stats = pd.DataFrame({
        'UserID': [unk_user_id],
        'user_rating_mean': [global_user_mean],
        'user_rating_std': [global_user_std],
        'user_rating_count': [global_user_count]
    })
    
    # 为UNK电影创建统计特征(使用全局平均值)
    global_movie_mean = movie_stats['movie_rating_mean'].mean()
    global_movie_std = movie_stats['movie_rating_std'].mean()
    global_movie_count = movie_stats['movie_rating_count'].mean()
    
    unk_movie_stats = pd.DataFrame({
        'MovieID': [unk_movie_id],
        'movie_rating_mean': [global_movie_mean],
        'movie_rating_std': [global_movie_std],
        'movie_rating_count': [global_movie_count]
    })
    
    # 将UNK统计特征添加到统计表中
    user_stats_with_unk = pd.concat([user_stats, unk_user_stats], ignore_index=True)
    movie_stats_with_unk = pd.concat([movie_stats, unk_movie_stats], ignore_index=True)
    
    return user_stats_with_unk, movie_stats_with_unk
