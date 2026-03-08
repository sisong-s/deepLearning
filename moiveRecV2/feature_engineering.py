"""
特征工程模块
负责从原始数据中提取和准备特征
"""

import numpy as np
import pandas as pd
import re
import os


def preprocess_users(users):
    """
    预处理用户数据
    
    Args:
        users: 原始用户DataFrame
    
    Returns:
        DataFrame: 处理后的用户数据
    """
    users = users.copy()
    
    # 1. UserID: 保留原样,转为int32
    users['UserID'] = users['UserID'].astype('int32')
    
    # 2. Gender: M→1, F→0
    users['Gender'] = users['Gender'].map({'M': 1, 'F': 0}).astype('int8')
    
    age_map = {val: i for i, val in enumerate(sorted(users['Age'].unique()))}
    users['Age'] = users['Age'].map(age_map)
    # 5. Zip-code: 删除
    users = users.drop(columns=['Zip-code'])
    
    print(f"  用户数据处理完成: {users.shape}")
    return users


def preprocess_movies(movies):
    """
    预处理电影数据
    
    Args:
        movies: 原始电影DataFrame
    
    Returns:
        DataFrame: 处理后的电影数据
    """
    movies = movies.copy()
    
    # 1. MovieID: 保留原样,转为int32
    movies['MovieID'] = movies['MovieID'].astype('int32')
    
    # 2. 提取年份
    movies['MoiveYear'] = movies['Title'].str.extract(r'\((\d{4})\)').astype('float')
    
    # 3. 清洗标题:去掉年份和多余空格
    movies['Title'] = movies['Title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()
    
    # 4. 处理Genres:切分成列表
    movies['Genres'] = movies['Genres'].str.split('|')
    
    print(f"  电影数据处理完成: {movies.shape}")
    return movies


def preprocess_ratings(ratings):
    """
    预处理评分数据
    
    Args:
        ratings: 原始评分DataFrame
    
    Returns:
        DataFrame: 处理后的评分数据
    """
    ratings = ratings.copy()
    
    # 1. UserID / MovieID: 保留,确保为int32
    ratings['UserID'] = ratings['UserID'].astype('int32')
    ratings['MovieID'] = ratings['MovieID'].astype('int32')
    
    # 2. Rating: 保留原样
    ratings['Rating'] = ratings['Rating'].astype('int8')
    
    # 3. Timestamp: 转为datetime并提取年月日
    ratings['DateTime'] = pd.to_datetime(ratings['Timestamp'], unit='s')
    ratings['RateYear'] = ratings['DateTime'].dt.year.astype('int16')
    ratings['Month'] = ratings['DateTime'].dt.month.astype('int8')
    ratings['Day'] = ratings['DateTime'].dt.day.astype('int8')
    # 保留原始Timestamp字段
    
    print(f"  评分数据处理完成: {ratings.shape}")
    return ratings


def augment_rating_data(train_data, target_distribution='uniform', augment_ratio=1.0):
    """
    数据增强:解决评分分布不均问题
    
    策略:对少数类评分(1,2,3)进行过采样,平衡评分分布
    
    Args:
        train_data: 训练集DataFrame
        target_distribution: 目标分布类型
            - 'uniform': 均匀分布(所有评分样本数相等)
            - 'balanced': 平衡分布(低分和高分样本数相当)
        augment_ratio: 增强比例 (0.0-1.0)
            - 1.0表示完全平衡
            - 0.5表示部分平衡(中间状态)
    
    Returns:
        DataFrame: 增强后的训练数据
    """
    print(f"=== 数据增强:解决评分分布不均 ===")
    print(f"  策略: {target_distribution} distribution")
    print(f"  增强比例: {augment_ratio*100:.0f}%")
    
    # 统计原始分布
    original_counts = train_data['Rating'].value_counts().sort_index()
    print(f"原始评分分布:")
    for rating, count in original_counts.items():
        print(f"    评分{rating}: {count:,} ({count/len(train_data)*100:.2f}%)")
    
    # 根据目标分布计算每个评分需要的样本数
    if target_distribution == 'uniform':
        # 均匀分布:所有评分样本数等于最大值
        max_count = original_counts.max()
        target_counts = {rating: max_count for rating in original_counts.index}
    
    elif target_distribution == 'balanced':
        # 平衡分布:低分(1-2)和高分(4-5)样本数相当
        low_ratings = original_counts[[1, 2]].sum()
        high_ratings = original_counts[[4, 5]].sum()
        target_high = max(low_ratings, high_ratings)
        
        target_counts = {
            1: int(target_high / 2),  # 低分评分1和2各占一半
            2: int(target_high / 2),
            3: original_counts[3],    # 中间评分保持不变
            4: original_counts[4],    # 高分保持不变
            5: original_counts[5]
        }
    
    else:
        raise ValueError(f"Unsupported target_distribution: {target_distribution}")
    
    # 应用增强比例
    augmented_data = [train_data]
    
    for rating in original_counts.index:
        current_count = original_counts[rating]
        target_count = target_counts[rating]
        
        # 计算需要增加的样本数
        samples_to_add = int((target_count - current_count) * augment_ratio)
        
        if samples_to_add > 0:
            # 从该评分的样本中随机采样(有放回)
            rating_samples = train_data[train_data['Rating'] == rating]
            augmented_samples = rating_samples.sample(n=samples_to_add, replace=True, random_state=42)
            augmented_data.append(augmented_samples)
            
            print(f"  评分{rating}: 增加 {samples_to_add:,} 个样本 ({current_count:,} → {current_count + samples_to_add:,})")
    
    # 合并原始数据和增强数据
    augmented_data = pd.concat(augmented_data, ignore_index=True)
    
    # 打乱顺序
    augmented_data = augmented_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 统计增强后的分布
    augmented_counts = augmented_data['Rating'].value_counts().sort_index()
    print(f"增强后评分分布:")
    for rating, count in augmented_counts.items():
        print(f"    评分{rating}: {count:,} ({count/len(augmented_data)*100:.2f}%)")
    
    print(f"数据集大小: {len(train_data):,} → {len(augmented_data):,} (+{len(augmented_data) - len(train_data):,}, +{(len(augmented_data)/len(train_data) - 1)*100:.1f}%)")
    print(f"  ==============================")
    
    return augmented_data


def calculate_user_stats(train_data):
    """
    计算用户行为统计特征 (隐式反馈版本: 只统计交互次数)
    
    Args:
        train_data: 训练集DataFrame
        
    Returns:
        DataFrame: 用户统计特征
    """
    user_stats = train_data.groupby('UserID').size().reset_index(name='user_interact_count')
    return user_stats


def calculate_movie_stats(train_data):
    """
    计算电影热度特征 (隐式反馈版本: 只统计交互次数)
    
    Args:
        train_data: 训练集DataFrame
        
    Returns:
        DataFrame: 电影统计特征
    """
    movie_stats = train_data.groupby('MovieID').size().reset_index(name='movie_interact_count')
    return movie_stats


def prepare_features(train_data, val_data, test_data, users=None, movies=None,
                     apply_augmentation=False, augment_config=None):
    """
    准备训练、验证和测试集的特征 (隐式反馈版本)

    标签: 1 表示用户对物品有过交互 (正例), 0 表示无交互 (负例, 由训练循环动态采样)
    测试评估: Leave-Last-1-Out, 每个用户有 1 个正例 + 100 个随机负例候选

    Args:
        train_data: 训练集 DataFrame
        val_data: 验证集 DataFrame
        test_data: 测试集 DataFrame
        users: 用户数据 DataFrame (可选)
        movies: 电影数据 DataFrame (可选)
        apply_augmentation: 保留参数, 隐式反馈场景不使用
        augment_config: 保留参数, 隐式反馈场景不使用

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test,
                feature_stats, test_neg_samples)
               test_neg_samples: dict {user_id -> list of 100 negative movie_ids}
    """
    print("[3/5] 准备特征 (隐式反馈模式)...")

    # 预处理评分数据
    train_data = preprocess_ratings(train_data)
    val_data   = preprocess_ratings(val_data)
    test_data  = preprocess_ratings(test_data)

    # ===== 构建全量交互集 (用于负采样) =====
    all_movie_ids = set(
        pd.concat([train_data['MovieID'], val_data['MovieID'], test_data['MovieID']]).unique()
    )

    # 每个用户在 train+val+test 中交互过的所有电影 (不应出现在负例中)
    user_interacted = (
        pd.concat([train_data[['UserID', 'MovieID']],
                   val_data[['UserID', 'MovieID']],
                   test_data[['UserID', 'MovieID']]])
        .groupby('UserID')['MovieID']
        .apply(set)
        .to_dict()
    )

    # ===== 计算统计特征 (交互次数) =====
    print("=== 计算统计特征 ===")
    user_stats  = calculate_user_stats(train_data)
    movie_stats = calculate_movie_stats(train_data)

    print(f"  用户统计特征: {len(user_stats)} 个用户")
    print(f"    - 平均交互次数: {user_stats['user_interact_count'].mean():.1f}")
    print(f"  电影统计特征: {len(movie_stats)} 部电影")
    print(f"    - 平均交互次数: {movie_stats['movie_interact_count'].mean():.1f}")
    print("======================\n")

    # ===== 合并用户/电影侧特征 =====
    if users is not None and movies is not None:
        users  = preprocess_users(users)
        movies = preprocess_movies(movies)

        train_data = train_data.merge(users, on='UserID', how='left').merge(movies, on='MovieID', how='left')
        val_data   = val_data.merge(users,   on='UserID', how='left').merge(movies, on='MovieID', how='left')
        test_data  = test_data.merge(users,  on='UserID', how='left').merge(movies, on='MovieID', how='left')

    # ===== 合并统计特征 =====
    train_data = train_data.merge(user_stats,  on='UserID',  how='left')
    val_data   = val_data.merge(user_stats,    on='UserID',  how='left')
    test_data  = test_data.merge(user_stats,   on='UserID',  how='left')

    train_data = train_data.merge(movie_stats, on='MovieID', how='left')
    val_data   = val_data.merge(movie_stats,   on='MovieID', how='left')
    test_data  = test_data.merge(movie_stats,  on='MovieID', how='left')

    print("  已合并用户行为统计特征和电影热度特征")

    # ===== 年份映射 =====
    all_data = pd.concat([train_data, val_data, test_data])

    movie_year_unique = sorted([y for y in all_data['MoiveYear'].dropna().unique()])
    movie_year_map    = {year: idx for idx, year in enumerate(movie_year_unique)}

    rate_year_unique  = sorted([y for y in all_data['RateYear'].dropna().unique()])
    rate_year_map     = {year: idx for idx, year in enumerate(rate_year_unique)}

    for df in [train_data, val_data, test_data]:
        df['MoiveYear'] = df['MoiveYear'].map(movie_year_map).fillna(0).astype(int)
        df['RateYear']  = df['RateYear'].map(rate_year_map).fillna(0).astype(int)

    # ===== 归一化统计特征 =====
    stats_cols = ['user_interact_count', 'movie_interact_count']
    print(f"=== 归一化统计特征 ===")
    for col in stats_cols:
        min_val = train_data[col].min()
        max_val = train_data[col].max()
        if max_val == min_val:
            for df in [train_data, val_data, test_data]:
                df[col] = 0.0
        else:
            for df in [train_data, val_data, test_data]:
                df[col] = ((df[col] - min_val) / (max_val - min_val)).clip(0, 1)
            print(f"  {col}: min={min_val:.1f}, max={max_val:.1f}")
    print("========================")

    # ===== 选择特征列 =====
    if users is not None and movies is not None:
        feature_cols = ['UserID', 'MovieID', 'Gender', 'Age', 'Occupation',
                        'MoiveYear', 'RateYear',
                        'user_interact_count', 'movie_interact_count']
    else:
        feature_cols = ['UserID', 'MovieID',
                        'user_interact_count', 'movie_interact_count']
    print(f"  使用特征: {', '.join(feature_cols)}")

    # ===== 特征统计信息 =====
    feature_stats = {}
    for col in feature_cols:
        if col == 'MoiveYear':
            unique_count = len(movie_year_map)
            max_val = unique_count - 1
            min_val = 0
        elif col == 'RateYear':
            unique_count = len(rate_year_map)
            max_val = unique_count - 1
            min_val = 0
        elif col == 'UserID':
            unique_values = train_data[col].dropna().unique()
            unique_count  = len(unique_values)
            max_val       = int(unique_values.max())
            min_val       = int(unique_values.min())
        elif col == 'MovieID':
            unique_values = train_data[col].dropna().unique()
            unique_count  = len(unique_values)
            max_val       = int(unique_values.max())
            min_val       = int(unique_values.min())
        else:
            combined      = pd.concat([train_data[col], val_data[col], test_data[col]])
            unique_values = combined.dropna().unique()
            unique_count  = len(unique_values)
            max_val       = int(unique_values.max())
            min_val       = int(unique_values.min())
        feature_stats[col] = {'num_unique': unique_count, 'max_value': max_val, 'min_value': min_val}

    print("=== 特征统计信息 ===")
    for col, stats in feature_stats.items():
        print(f"  {col}: {stats['num_unique']} 个唯一值 (范围: {stats['min_value']}-{stats['max_value']})")
    print("  =====================")

    # ===== 构建 X/y =====
    X_train = train_data[feature_cols].values.astype(np.float32)
    y_train = np.ones(len(X_train), dtype=np.float32)   # 全部为正例 (负采样在训练中动态完成)

    X_val   = val_data[feature_cols].values.astype(np.float32)
    y_val   = np.ones(len(X_val), dtype=np.float32)

    X_test  = test_data[feature_cols].values.astype(np.float32)
    y_test  = np.ones(len(X_test), dtype=np.float32)

    print(f"特征维度: {X_train.shape[1]}")
    print(f"  训练集正例数: {len(X_train):,}")
    print(f"  验证集正例数: {len(X_val):,}")
    print(f"  测试集正例数: {len(X_test):,}")

    # ===== 构建测试负采样候选池 (每个用户 100 个负例) =====
    print("\n=== 构建测试负采样候选池 (每用户 100 个负例) ===")
    rng = np.random.default_rng(42)
    all_movie_list = sorted(all_movie_ids)
    test_neg_samples = {}   # {user_id -> array of 100 negative movie_ids}

    for user_id in test_data['UserID'].unique():
        interacted = user_interacted.get(user_id, set())
        candidates = [m for m in all_movie_list if m not in interacted]
        if len(candidates) >= 100:
            neg_movies = rng.choice(candidates, size=100, replace=False).tolist()
        else:
            neg_movies = candidates  # 极少情况下不足100个
        test_neg_samples[user_id] = neg_movies

    print(f"  已为 {len(test_neg_samples)} 个测试用户构建负例候选池")
    print("  =====================")

    # ===== 保存中间结果 =====
    os.makedirs('result', exist_ok=True)
    train_data.to_csv('result/train_data.csv', index=False, encoding='utf-8-sig')
    val_data.to_csv('result/val_data.csv',     index=False, encoding='utf-8-sig')
    test_data.to_csv('result/test_data.csv',   index=False, encoding='utf-8-sig')
    print(f"  数据已保存到 result/ 目录")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_stats, test_neg_samples



