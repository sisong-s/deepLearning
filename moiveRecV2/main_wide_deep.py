"""
MovieLens 1M - Wide&Deep 隐式反馈版本
数据集划分: 基于用户的 Leave-Last-1-Out (train / val / test)
标签:       隐式信号 0/1 (有无交互)
训练:       负采样 (1正:4负) + BCE 损失
评估:       HR@10, NDCG@10 (每用户 100 个随机负例)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from data_pipeline import load_data, split_data_by_user
from feature_engineering import (
    prepare_features, preprocess_users, preprocess_movies,
    preprocess_ratings, calculate_user_stats, calculate_movie_stats
)
from model_wide_deep import train_wide_deep_model, WideAndDeepModel
from evaluation import evaluate_model


def main():
    print("=" * 80)
    print("MovieLens 1M - Wide&Deep 隐式反馈版本 (BCE + 负采样)")
    print("=" * 80)

    # 1. 加载数据
    ratings, users, movies = load_data()

    # 2. 按用户划分数据集 (Leave-Last-1-Out)
    train_data, val_data, test_data = split_data_by_user(ratings)

    # 3. 准备特征
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     feature_stats,
     test_neg_samples) = prepare_features(
        train_data, val_data, test_data, users, movies,
        apply_augmentation=False
    )

    # 从 feature_stats 提取模型参数
    num_users       = feature_stats['UserID']['max_value']
    num_movies      = feature_stats['MovieID']['max_value']
    num_ages        = feature_stats['Age']['num_unique']
    num_occupations = feature_stats['Occupation']['num_unique']
    num_movie_years = feature_stats['MoiveYear']['num_unique']
    num_rate_years  = feature_stats['RateYear']['num_unique']
    num_stat_features = 2   # user_interact_count + movie_interact_count

    print(f"\n模型参数配置:")
    print(f"  num_users: {num_users} (含UNK)")
    print(f"  num_movies: {num_movies} (含UNK)")
    print(f"  num_ages: {num_ages}")
    print(f"  num_occupations: {num_occupations}")
    print(f"  num_movie_years: {num_movie_years}")
    print(f"  num_rate_years: {num_rate_years}")
    print(f"  num_stat_features: {num_stat_features}")

    # 4. 训练或加载模型
    model_path            = 'models/wide_deep_implicit.pth'
    hidden_units          = [64, 32]
    dropout_rate          = 0.3
    wide_l2_reg           = 0.01
    early_stopping_patience   = 3
    early_stopping_min_delta  = 0.0001

    # 收集全量电影ID (供负采样使用)
    all_movie_ids = list(set(ratings['MovieID'].unique()))

    wd_model = WideAndDeepModel.load(
        model_path,
        hidden_units=hidden_units,
        num_users=num_users,
        num_movies=num_movies,
        num_ages=num_ages,
        num_occupations=num_occupations,
        num_movie_years=num_movie_years,
        num_rate_years=num_rate_years,
        num_stat_features=num_stat_features,
        dropout_rate=dropout_rate,
        wide_l2_reg=wide_l2_reg
    )

    if wd_model is None:
        wd_model = train_wide_deep_model(
            X_train, y_train,
            X_val,   y_val,
            hidden_units=hidden_units,
            epochs=30,
            batch_size=512,
            learning_rate=0.001,
            verbose=True,
            num_users=num_users,
            num_movies=num_movies,
            num_ages=num_ages,
            num_occupations=num_occupations,
            num_movie_years=num_movie_years,
            num_rate_years=num_rate_years,
            num_stat_features=num_stat_features,
            dropout_rate=dropout_rate,
            wide_l2_reg=wide_l2_reg,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            neg_sample_ratio=4,
            all_movie_ids=all_movie_ids
        )
        wd_model.save(model_path)
    else:
        print(f"[4/5] 使用已保存的模型: {model_path}")

    # 5. 评估: 需要重建评估所需的辅助数据结构
    # 重新获取预处理后的 users/movies/stats (用于评估时构造特征)
    _users  = preprocess_users(users.copy())
    _movies = preprocess_movies(movies.copy())

    # 重新计算 train 统计 (与 prepare_features 一致)
    _train_raw = preprocess_ratings(train_data.copy())
    _user_stats  = calculate_user_stats(_train_raw)
    _movie_stats = calculate_movie_stats(_train_raw)

    # 重建年份映射 (与 prepare_features 一致)
    _all = pd.concat([
        preprocess_ratings(train_data.copy()),
        preprocess_ratings(val_data.copy()),
        preprocess_ratings(test_data.copy())
    ])
    # 注意: MoiveYear 来自 movies 表, 需合并后才能获得; 这里借助 _movies 表查询原始年份
    # evaluation 中 get_movie_features 直接用 movies_df 的 MoiveYear (原始年份) 去查 movie_year_map
    # movie_year_map 由 prepare_features 构建时基于合并后数据集的 MoiveYear 字段
    # 此处重建: 合并 _movies 取所有电影年份
    _all_movie_years = sorted(_movies['MoiveYear'].dropna().unique())
    movie_year_map = {y: i for i, y in enumerate(_all_movie_years)}
    rate_year_map  = {y: i for i, y in enumerate(sorted(_all['RateYear'].dropna().unique()))}

    # 重建测试集带侧面特征的 DataFrame
    _test_raw = preprocess_ratings(test_data.copy())
    _test_raw = _test_raw.merge(_users,  on='UserID',  how='left')
    _test_raw = _test_raw.merge(_movies, on='MovieID', how='left')

    # 添加 UNK 行到统计表 (保持与训练一致)
    max_uid = _train_raw['UserID'].max()
    max_mid = _train_raw['MovieID'].max()
    _user_stats  = pd.concat([_user_stats,
                               pd.DataFrame({'UserID': [max_uid+1],
                                             'user_interact_count': [_user_stats['user_interact_count'].mean()]})],
                              ignore_index=True)
    _movie_stats = pd.concat([_movie_stats,
                               pd.DataFrame({'MovieID': [max_mid+1],
                                             'movie_interact_count': [_movie_stats['movie_interact_count'].mean()]})],
                              ignore_index=True)

    feature_cols = ['UserID', 'MovieID', 'Gender', 'Age', 'Occupation',
                    'MoiveYear', 'RateYear',
                    'user_interact_count', 'movie_interact_count']

    metrics = evaluate_model(
        model=wd_model,
        test_data=_test_raw,
        test_neg_samples=test_neg_samples,
        feature_cols=feature_cols,
        users_df=_users,
        movies_df=_movies,
        movie_year_map=movie_year_map,
        rate_year_map=rate_year_map,
        user_stats=_user_stats,
        movie_stats=_movie_stats,
        k=10
    )

    print("\n" + "=" * 80)
    print("实验完成!")
    print(f"  HR@10:   {metrics['hr']:.4f}")
    print(f"  NDCG@10: {metrics['ndcg']:.4f}")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    main()
