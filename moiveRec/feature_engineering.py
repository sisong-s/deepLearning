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
    计算用户行为统计特征
    
    Args:
        train_data: 训练集DataFrame
        
    Returns:
        DataFrame: 用户统计特征
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
    """
    movie_stats = train_data.groupby('MovieID')['Rating'].agg([
        ('movie_rating_mean', 'mean'),     # 电影平均评分
        ('movie_rating_std', 'std'),       # 电影评分标准差
        ('movie_rating_count', 'count')    # 电影评分总数(热度)
    ]).reset_index()
    
    # 填充标准差的NaN值(只有1次评分的电影)
    movie_stats['movie_rating_std'] = movie_stats['movie_rating_std'].fillna(0)
    
    return movie_stats


def prepare_features(train_data, val_data, test_data, users=None, movies=None, 
                    apply_augmentation=False, augment_config=None):
    """
    准备训练、验证和测试集的特征
    
    Args:
        train_data: 训练集DataFrame
        val_data: 验证集DataFrame
        test_data: 测试集DataFrame
        users: 用户数据DataFrame (可选)
        movies: 电影数据DataFrame (可选)
        apply_augmentation: 是否应用数据增强
        augment_config: 数据增强配置字典
            - target_distribution: 'uniform' 或 'balanced'
            - augment_ratio: 0.0-1.0
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, feature_stats)
    """
    print("[3/5] 准备特征...")
    
    # 预处理评分数据
    train_data = preprocess_ratings(train_data)
    val_data = preprocess_ratings(val_data)
    test_data = preprocess_ratings(test_data)
    
    # ===== 处理OOV (Out-Of-Vocabulary) 用户和电影 =====
    # 获取训练集中的用户和电影ID集合
    train_user_ids = set(train_data['UserID'].unique())
    train_movie_ids = set(train_data['MovieID'].unique())
    
    # 统计OOV数量
    val_oov_users = set(val_data['UserID'].unique()) - train_user_ids
    val_oov_movies = set(val_data['MovieID'].unique()) - train_movie_ids
    test_oov_users = set(test_data['UserID'].unique()) - train_user_ids
    test_oov_movies = set(test_data['MovieID'].unique()) - train_movie_ids
    
    print(f"=== OOV (未见过的ID) 统计 ===")
    print(f"  训练集: {len(train_user_ids)} 个用户, {len(train_movie_ids)} 部电影")
    print(f"  验证集OOV: {len(val_oov_users)} 个新用户, {len(val_oov_movies)} 部新电影")
    print(f"  测试集OOV: {len(test_oov_users)} 个新用户, {len(test_oov_movies)} 部新电影")
    
    # 定义UNK索引 (使用最大ID+1作为UNK索引)
    # 注意:这里的UNK索引会在模型中对应一个可学习的embedding向量
    max_user_id = train_data['UserID'].max()
    max_movie_id = train_data['MovieID'].max()
    UNK_USER_ID = max_user_id + 1
    UNK_MOVIE_ID = max_movie_id + 1
    
    print(f"  UNK标记: UserID={UNK_USER_ID}, MovieID={UNK_MOVIE_ID}")
    print(f"  ===========================")
    
    # === 验证集和测试集:先将OOV ID替换为UNK,再merge特征 ===
    # 这样OOV样本会使用统一的UNK特征,而不是原始ID的特征
    val_data.loc[~val_data['UserID'].isin(train_user_ids), 'UserID'] = UNK_USER_ID
    val_data.loc[~val_data['MovieID'].isin(train_movie_ids), 'MovieID'] = UNK_MOVIE_ID
    
    test_data.loc[~test_data['UserID'].isin(train_user_ids), 'UserID'] = UNK_USER_ID
    test_data.loc[~test_data['MovieID'].isin(train_movie_ids), 'MovieID'] = UNK_MOVIE_ID
    
    # === 计算用户行为统计特征和电影热度特征 ===
    print("=== 计算统计特征 ===")
    
    user_stats = calculate_user_stats(train_data)
    movie_stats = calculate_movie_stats(train_data)
    
    print(f"  用户统计特征: {len(user_stats)} 个用户")
    print(f"    - 平均评分均值: {user_stats['user_rating_mean'].mean():.3f}")
    print(f"    - 平均评分数: {user_stats['user_rating_count'].mean():.1f}")
    
    print(f"  电影统计特征: {len(movie_stats)} 部电影")
    print(f"    - 平均评分均值: {movie_stats['movie_rating_mean'].mean():.3f}")
    print(f"    - 平均评分数: {movie_stats['movie_rating_count'].mean():.1f}")
    print("======================")
    
    # === 数据增强:在特征合并之前进行(保证OOV处理的一致性) ===
    if apply_augmentation:
        if augment_config is None:
            augment_config = {'target_distribution': 'uniform', 'augment_ratio': 0.5}
        train_data = augment_rating_data(train_data, **augment_config)
        
        # 数据增强后重新计算统计特征
        print("=== 数据增强后重新计算统计特征 ===")
        user_stats = calculate_user_stats(train_data)
        movie_stats = calculate_movie_stats(train_data)
        print(f"  用户统计特征已更新: {len(user_stats)} 个用户")
        print(f"  电影统计特征已更新: {len(movie_stats)} 部电影")
    print("======================\n")
    
    # 如果提供了用户和电影数据,进行合并
    if users is not None and movies is not None:
        users = preprocess_users(users)
        movies = preprocess_movies(movies)
        
        # === 为UNK ID创建特征行 ===
        # 创建UNK用户特征:使用训练集中所有用户特征的众数(最常见值)
        unk_user_features = pd.DataFrame({
            'UserID': [UNK_USER_ID],
            'Gender': [users['Gender'].mode()[0]],  # 众数
            'Age': [users['Age'].mode()[0]],
            'Occupation': [users['Occupation'].mode()[0]]
        })
        
        # 创建UNK电影特征:使用训练集中所有电影特征的众数
        unk_movie_features = pd.DataFrame({
            'MovieID': [UNK_MOVIE_ID],
            'Title': ['Unknown'],
            'MoiveYear': [movies['MoiveYear'].mode()[0]],  # 众数年份
            'Genres': [movies['Genres'].mode()[0]]  # 众数类型
        })
        
        # 将UNK特征添加到users和movies表中
        users = pd.concat([users, unk_user_features], ignore_index=True)
        movies = pd.concat([movies, unk_movie_features], ignore_index=True)
        
        print(f"  已为UNK ID创建特征行 (UserID={UNK_USER_ID}, MovieID={UNK_MOVIE_ID})")
        
        # 合并特征 (此时val/test中的OOV ID已经替换为UNK_USER_ID/UNK_MOVIE_ID)
        train_data = train_data.merge(users, on='UserID', how='left')
        train_data = train_data.merge(movies, on='MovieID', how='left')
        
        val_data = val_data.merge(users, on='UserID', how='left')
        val_data = val_data.merge(movies, on='MovieID', how='left')
        
        test_data = test_data.merge(users, on='UserID', how='left')
        test_data = test_data.merge(movies, on='MovieID', how='left')
    
    # === 合并统计特征 ===
    # 为UNK用户和电影创建统计特征(使用全局平均值)
    global_mean = user_stats['user_rating_mean'].mean()
    global_std = user_stats['user_rating_std'].mean()
    global_count_user = user_stats['user_rating_count'].mean()
    
    unk_user_stats = pd.DataFrame({
        'UserID': [UNK_USER_ID],
        'user_rating_mean': [global_mean],
        'user_rating_std': [global_std],
        'user_rating_count': [global_count_user]
    })
    
    global_movie_mean = movie_stats['movie_rating_mean'].mean()
    global_movie_std = movie_stats['movie_rating_std'].mean()
    global_count_movie = movie_stats['movie_rating_count'].mean()
    
    unk_movie_stats = pd.DataFrame({
        'MovieID': [UNK_MOVIE_ID],
        'movie_rating_mean': [global_movie_mean],
        'movie_rating_std': [global_movie_std],
        'movie_rating_count': [global_count_movie]
    })
    
    # 将UNK统计特征添加到统计表中
    user_stats = pd.concat([user_stats, unk_user_stats], ignore_index=True)
    movie_stats = pd.concat([movie_stats, unk_movie_stats], ignore_index=True)
    
    # 合并用户统计特征
    train_data = train_data.merge(user_stats, on='UserID', how='left')
    val_data = val_data.merge(user_stats, on='UserID', how='left')
    test_data = test_data.merge(user_stats, on='UserID', how='left')
    
    # 合并电影统计特征
    train_data = train_data.merge(movie_stats, on='MovieID', how='left')
    val_data = val_data.merge(movie_stats, on='MovieID', how='left')
    test_data = test_data.merge(movie_stats, on='MovieID', how='left')
    
    print("  已合并用户行为统计特征和电影热度特征")
    
    # === 训练集:复制部分样本并将ID设为UNK,让模型学习UNK embedding ===
    # 优点:保留原始训练数据,UNK embedding会学到"平均用户/电影"的表示
    # 注意:需要在合并用户和电影特征之后进行,这样UNK样本才能包含完整特征
    np.random.seed(42)
    unk_ratio = 0.05  # 随机复制5%的样本用于UNK学习
    
    n_train = len(train_data)
    
    # 随机选择5%的样本,复制并将UserID设为UNK
    unk_user_indices = np.random.choice(train_data.index, size=int(n_train * unk_ratio), replace=False)
    unk_user_samples = train_data.loc[unk_user_indices].copy()
    unk_user_samples['UserID'] = UNK_USER_ID
    
    # 随机选择5%的样本,复制并将MovieID设为UNK
    unk_movie_indices = np.random.choice(train_data.index, size=int(n_train * unk_ratio), replace=False)
    unk_movie_samples = train_data.loc[unk_movie_indices].copy()
    unk_movie_samples['MovieID'] = UNK_MOVIE_ID
    
    # 将UNK样本追加到训练集
    train_data = pd.concat([train_data, unk_user_samples, unk_movie_samples], ignore_index=True)
    
    print(f"  训练集UNK增强: 添加 {len(unk_user_samples)} 个UserID=UNK样本, {len(unk_movie_samples)} 个MovieID=UNK样本")
    print(f"  增强后训练集大小: {len(train_data)} (原始: {n_train}, 增加: {len(unk_user_samples) + len(unk_movie_samples)})")
    
    # 合并所有数据以统一映射
    all_data = pd.concat([train_data, val_data, test_data])
    
    # 创建年份映射 (MoiveYear和RateYear)
    # MoiveYear: 将实际年份映射为索引
    movie_year_unique = sorted([y for y in all_data['MoiveYear'].dropna().unique()])
    movie_year_map = {year: idx for idx, year in enumerate(movie_year_unique)}
    
    # RateYear: 将实际年份映射为索引  
    rate_year_unique = sorted([y for y in all_data['RateYear'].dropna().unique()])
    rate_year_map = {year: idx for idx, year in enumerate(rate_year_unique)}
    
    # 应用映射
    train_data['MoiveYear'] = train_data['MoiveYear'].map(movie_year_map).fillna(0).astype(int)
    train_data['RateYear'] = train_data['RateYear'].map(rate_year_map).fillna(0).astype(int)
    
    val_data['MoiveYear'] = val_data['MoiveYear'].map(movie_year_map).fillna(0).astype(int)
    val_data['RateYear'] = val_data['RateYear'].map(rate_year_map).fillna(0).astype(int)
    
    test_data['MoiveYear'] = test_data['MoiveYear'].map(movie_year_map).fillna(0).astype(int)
    test_data['RateYear'] = test_data['RateYear'].map(rate_year_map).fillna(0).astype(int)
    
    # === 归一化统计特征 ===
    # 对std和count进行Min-Max归一化(0-1),不归一化mean
    stats_cols = ['user_rating_std', 'user_rating_count',
                  'movie_rating_std', 'movie_rating_count']
    
    print(f"=== 归一化统计特征 ===")
    normalization_stats = {}  # 保存归一化参数
    
    for col in stats_cols:
        # 使用训练集计算Min-Max归一化参数
        min_val = train_data[col].min()
        max_val = train_data[col].max()
        
        # 避免除以0
        if max_val == min_val:
            # 如果最大值等于最小值，设置为0
            train_data[col] = 0
            val_data[col] = 0
            test_data[col] = 0
        else:
            # 保存归一化参数
            normalization_stats[col] = {'min': min_val, 'max': max_val}
            
            # 应用Min-Max归一化: (x - min) / (max - min)
            train_data[col] = (train_data[col] - min_val) / (max_val - min_val)
            val_data[col] = (val_data[col] - min_val) / (max_val - min_val)
            test_data[col] = (test_data[col] - min_val) / (max_val - min_val)
            
            # 将超出范围的值裁剪到[0, 1]
            train_data[col] = train_data[col].clip(0, 1)
            val_data[col] = val_data[col].clip(0, 1)
            test_data[col] = test_data[col].clip(0, 1)
            
            print(f"  {col}: min={min_val:.3f}, max={max_val:.3f}")
    
    print(f"  统计特征已归一化 (Min-Max 0-1归一化, 不含mean字段)")
    print("========================")
    
    # 保存处理后的train_data、val_data、test_data到result文件夹
    os.makedirs('result', exist_ok=True)
    train_data.to_csv('result/train_data.csv', index=False, encoding='utf-8-sig')
    val_data.to_csv('result/val_data.csv', index=False, encoding='utf-8-sig')
    test_data.to_csv('result/test_data.csv', index=False, encoding='utf-8-sig')
    print(f"  训练数据已保存到: result/train_data.csv")
    print(f"  验证数据已保存到: result/val_data.csv")
    print(f"  测试数据已保存到: result/test_data.csv")
    # 选择特征列
    if users is not None and movies is not None:
        # 使用丰富特征:用户特征 + 电影特征 + 时间特征 + 统计特征
        feature_cols = ['UserID', 'MovieID', 'Gender', 'Age', 'Occupation', 
                       'MoiveYear', 'RateYear',
                       'user_rating_mean', 'user_rating_std', 'user_rating_count',
                       'movie_rating_mean', 'movie_rating_std', 'movie_rating_count']
        print(f"  使用特征: {', '.join(feature_cols)}")
    else:
        # 只使用ID特征 + 统计特征
        feature_cols = ['UserID', 'MovieID',
                       'user_rating_mean', 'user_rating_std', 'user_rating_count',
                       'movie_rating_mean', 'movie_rating_std', 'movie_rating_count']
        print(f"  使用特征: {', '.join(feature_cols)}")
    # 统计类别特征的唯一值数量
    feature_stats = {}
    # 对每个特征统计唯一值 (使用映射后的数据)
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
            # 训练集已包含UNK样本,直接统计
            unique_values = train_data[col].dropna().unique()
            unique_count = len(unique_values)  # 已包含UNK
            max_val = int(unique_values.max())  # 已经是UNK_USER_ID
            min_val = int(unique_values.min())
        elif col == 'MovieID':
            # 训练集已包含UNK样本,直接统计
            unique_values = train_data[col].dropna().unique()
            unique_count = len(unique_values)  # 已包含UNK
            max_val = int(unique_values.max())  # 已经是UNK_MOVIE_ID
            min_val = int(unique_values.min())
        else:
            combined = pd.concat([train_data[col], val_data[col], test_data[col]])
            unique_values = combined.dropna().unique()
            unique_count = len(unique_values)
            max_val = int(unique_values.max())
            min_val = int(unique_values.min())
        feature_stats[col] = {
            'num_unique': unique_count,
            'max_value': max_val,
            'min_value': min_val
        }
    print("=== 特征统计信息 ===")
    for col, stats in feature_stats.items():
        print(f"  {col}: {stats['num_unique']} 个唯一值 (范围: {stats['min_value']}-{stats['max_value']})")
    print("  =====================")
    X_train = train_data[feature_cols].values
    y_train = train_data['Rating'].values
    X_val = val_data[feature_cols].values
    y_val = val_data['Rating'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['Rating'].values
    print(f"特征维度: {X_train.shape[1]}")
    print(f"  训练集样本数: {len(X_train):,}")
    print(f"  验证集样本数: {len(X_val):,}")
    print(f"  测试集样本数: {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_stats

def convert_to_classification(y_train, y_val, y_test):

    """
    将评分从1-5映射到0-4,用于分类任务
    Args:
        y_train, y_val, y_test: 原始评分数组
    Returns:
        tuple: (y_train_class, y_val_class, y_test_class)
    """
    y_train_class = (y_train - 1).astype(int)
    y_val_class = (y_val - 1).astype(int)
    y_test_class = (y_test - 1).astype(int)
    return y_train_class, y_val_class, y_test_class

def proba_to_rating(y_proba):
    """
    将预测概率转换为期望评分
    Args:
        y_proba: 预测概率矩阵 (n_samples, 5)
    Returns:
        array: 期望评分 (概率加权)
    """
    # 计算期望评分 (概率加权)
    # 例如: np.dot([0.1, 0.2, 0.3, 0.25, 0.15], [1, 2, 3, 4, 5])
    #      = 0.1×1 + 0.2×2 + 0.3×3 + 0.25×4 + 0.15×5 = 3.15
    return np.dot(y_proba, np.array([1, 2, 3, 4, 5]))

