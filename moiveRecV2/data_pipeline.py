"""
数据管道模块
负责数据加载和按时间划分数据集
"""

import pandas as pd


def load_data():
    """
    加载MovieLens 1M数据集
    
    Returns:
        tuple: (ratings, users, movies) 三个DataFrame
    """
    print("\n[1/5] 加载数据...")
    
    # 加载评分数据
    ratings = pd.read_csv(
        'data/ratings.dat',
        sep='::',
        engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        encoding='utf-8'
    )
    
    # 加载用户数据
    users = pd.read_csv(
        'data/users.dat',
        sep='::',
        engine='python',
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
        encoding='utf-8'
    )
    
    # 加载电影数据
    movies = pd.read_csv(
        'data/movies.dat',
        sep='::',
        engine='python',
        names=['MovieID', 'Title', 'Genres'],
        encoding='latin-1'
    )
    
    print(f"  总评分记录数: {len(ratings):,}")
    print(f"  总用户数: {len(users):,}")
    print(f"  总电影数: {len(movies):,}")
    
    return ratings, users, movies


def split_data_by_time(ratings, train_ratio=0.8, val_ratio=0.1):
    """
    按时间划分数据集
    
    Args:
        ratings: 评分数据DataFrame
        train_ratio: 训练集比例 (默认0.8)
        val_ratio: 验证集比例 (默认0.1)
    
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    print(f"\n[2/5] 按时间划分数据集 (前{train_ratio*100:.0f}% train, "
          f"{train_ratio*100:.0f}%-{(train_ratio+val_ratio)*100:.0f}% val, "
          f"后{(1-train_ratio-val_ratio)*100:.0f}% test)...")
    
    # 按时间排序
    data_sorted = ratings.sort_values('Timestamp').reset_index(drop=True)
    
    n = len(data_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data_sorted[:train_end].copy()
    val_data = data_sorted[train_end:val_end].copy()
    test_data = data_sorted[val_end:].copy()
    
    print(f"  Train: {len(train_data):,} ({len(train_data)/n*100:.1f}%)")
    print(f"  Val:   {len(val_data):,} ({len(val_data)/n*100:.1f}%)")
    print(f"  Test:  {len(test_data):,} ({len(test_data)/n*100:.1f}%)")
    
    return train_data, val_data, test_data


def split_data_by_user(ratings):
    """
    以用户为单位划分数据集（Leave-Last-2-Out策略）
    每个用户按时间排序后，最后一条记录作为测试集，倒数第二条作为验证集，其余作为训练集。
    
    Args:
        ratings: 评分数据DataFrame，需包含 UserID 和 Timestamp 列
    
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    print("\n[2/5] 以用户为单位划分数据集 (Leave-Last-2-Out)...")

    # 按用户和时间排序，同一时间戳内保持稳定顺序
    data_sorted = ratings.sort_values(['UserID', 'Timestamp']).reset_index(drop=True)

    # 为每条记录计算其在该用户内的逆序排名（1=最后一条，2=倒数第二条，...）
    data_sorted['_rank_from_last'] = (
        data_sorted.groupby('UserID').cumcount(ascending=False) + 1
    )

    test_data  = data_sorted[data_sorted['_rank_from_last'] == 1].drop(columns='_rank_from_last').copy()
    val_data   = data_sorted[data_sorted['_rank_from_last'] == 2].drop(columns='_rank_from_last').copy()
    train_data = data_sorted[data_sorted['_rank_from_last'] >  2].drop(columns='_rank_from_last').copy()

    n = len(data_sorted)
    print(f"  Train: {len(train_data):,} ({len(train_data)/n*100:.1f}%)")
    print(f"  Val:   {len(val_data):,} ({len(val_data)/n*100:.1f}%)")
    print(f"  Test:  {len(test_data):,} ({len(test_data)/n*100:.1f}%)")

    return train_data, val_data, test_data

