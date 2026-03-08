"""
评估模块 (隐式反馈版本)
评估指标: HR@10, NDCG@10
评估方式: Leave-Last-1-Out + 每用户 100 个随机负例候选
"""

import numpy as np
import pandas as pd


def calculate_hr_and_ndcg_at_k(model, test_data, test_neg_samples, feature_template,
                                users_df, movies_df, movie_year_map, rate_year_map,
                                user_stats, movie_stats, k=10):
    """
    计算 HR@K 和 NDCG@K

    评估方法:
        对每个测试用户, 将其测试正例与 100 个随机负例组成候选集 (共 101 个),
        用模型预测所有候选的概率并排序, 计算正例是否在 Top-K 内.

    Args:
        model: 训练好的 WideAndDeepModel
        test_data: 测试集 DataFrame (每用户1条正例, 含 UserID/MovieID 及侧面特征)
        test_neg_samples: dict {user_id -> list of 100 negative movie_ids}
        feature_template: 特征列名列表 (与训练时一致)
        users_df: 预处理后的用户 DataFrame (含 Gender/Age/Occupation)
        movies_df: 预处理后的电影 DataFrame (含 MoiveYear)
        movie_year_map: 电影年份 -> 索引 的映射字典
        rate_year_map: 评分年份 -> 索引 的映射字典
        user_stats: 用户统计 DataFrame (UserID, user_interact_count)
        movie_stats: 电影统计 DataFrame (MovieID, movie_interact_count)
        k: Top-K (默认 10)

    Returns:
        tuple: (hr, ndcg) 平均 HR@K 和 NDCG@K
    """
    hr_list   = []
    ndcg_list = []

    # 归一化参数 (与 feature_engineering 保持一致: 用训练集统计)
    user_cnt_min  = user_stats['user_interact_count'].min()
    user_cnt_max  = user_stats['user_interact_count'].max()
    movie_cnt_min = movie_stats['movie_interact_count'].min()
    movie_cnt_max = movie_stats['movie_interact_count'].max()

    def norm_user_cnt(v):
        if user_cnt_max == user_cnt_min:
            return 0.0
        return float(np.clip((v - user_cnt_min) / (user_cnt_max - user_cnt_min), 0, 1))

    def norm_movie_cnt(v):
        if movie_cnt_max == movie_cnt_min:
            return 0.0
        return float(np.clip((v - movie_cnt_min) / (movie_cnt_max - movie_cnt_min), 0, 1))

    # 建立快速查询索引
    users_idx  = users_df.set_index('UserID')
    movies_idx = movies_df.set_index('MovieID')
    user_stat_idx  = user_stats.set_index('UserID')
    movie_stat_idx = movie_stats.set_index('MovieID')

    UNK_USER_ID  = user_stats['UserID'].max()   # 已包含 UNK 行
    UNK_MOVIE_ID = movie_stats['MovieID'].max()

    def get_user_features(uid):
        row = users_idx.loc[uid] if uid in users_idx.index else users_idx.loc[UNK_USER_ID]
        return int(row['Gender']), int(row['Age']), int(row['Occupation'])

    def get_movie_features(mid):
        row = movies_idx.loc[mid] if mid in movies_idx.index else movies_idx.loc[UNK_MOVIE_ID]
        raw_year = row['MoiveYear']
        return int(movie_year_map.get(raw_year, 0))

    def get_user_cnt(uid):
        if uid in user_stat_idx.index:
            return norm_user_cnt(user_stat_idx.loc[uid, 'user_interact_count'])
        return norm_user_cnt(user_stat_idx.loc[UNK_USER_ID, 'user_interact_count'])

    def get_movie_cnt(mid):
        if mid in movie_stat_idx.index:
            return norm_movie_cnt(movie_stat_idx.loc[mid, 'movie_interact_count'])
        return norm_movie_cnt(movie_stat_idx.loc[UNK_MOVIE_ID, 'movie_interact_count'])

    # 评分年份: 测试集中的 RateYear (固定为测试样本的年份)
    test_idx = test_data.set_index('UserID')

    for _, row in test_data.iterrows():
        user_id    = int(row['UserID'])
        pos_movie  = int(row['MovieID'])
        neg_movies = test_neg_samples.get(user_id, [])

        if len(neg_movies) == 0:
            continue

        # 候选列表: [正例] + [100 个负例]
        candidates = [pos_movie] + list(neg_movies)

        # 获取用户侧特征
        gender, age, occupation = get_user_features(user_id)
        rate_year = int(rate_year_map.get(int(row.get('RateYear', 0)), 0))
        u_cnt     = get_user_cnt(user_id)

        # 构建特征矩阵
        rows = []
        for mid in candidates:
            m_year = get_movie_features(mid)
            m_cnt  = get_movie_cnt(mid)
            feat = [user_id, mid, gender, age, occupation, m_year, rate_year, u_cnt, m_cnt]
            rows.append(feat)

        X_cand = np.array(rows, dtype=np.float32)
        scores = model.predict(X_cand)   # (101,)

        # 排序: 按分数降序
        ranked_indices = np.argsort(-scores)
        top_k_indices  = ranked_indices[:k]

        # 正例在索引 0
        hit    = 1 if 0 in top_k_indices else 0
        hr_list.append(hit)

        # NDCG@K: 正例命中时的折损
        if hit:
            rank = int(np.where(ranked_indices == 0)[0][0]) + 1  # 1-indexed
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            ndcg = 0.0
        ndcg_list.append(ndcg)

    hr   = float(np.mean(hr_list))   if hr_list   else 0.0
    ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    return hr, ndcg


def evaluate_model(model, test_data, test_neg_samples,
                   feature_cols, users_df, movies_df,
                   movie_year_map, rate_year_map,
                   user_stats, movie_stats, k=10):
    """
    全面评估模型性能 (隐式反馈: HR@K + NDCG@K)

    Args:
        model: 训练好的 WideAndDeepModel
        test_data: 测试集 DataFrame
        test_neg_samples: dict {user_id -> list of 100 neg movie_ids}
        feature_cols: 特征列名列表
        users_df: 预处理后的用户 DataFrame
        movies_df: 预处理后的电影 DataFrame
        movie_year_map: 电影年份映射
        rate_year_map: 评分年份映射
        user_stats: 用户交互统计 DataFrame
        movie_stats: 电影交互统计 DataFrame
        k: Top-K (默认 10)

    Returns:
        dict: {'hr': float, 'ndcg': float}
    """
    print(f"\n[5/5] 评估模型 (HR@{k} + NDCG@{k})...")
    print(f"  评估方式: Leave-Last-1-Out + 100 随机负例")
    print(f"  测试用户数: {test_data['UserID'].nunique()}")

    hr, ndcg = calculate_hr_and_ndcg_at_k(
        model=model,
        test_data=test_data,
        test_neg_samples=test_neg_samples,
        feature_template=feature_cols,
        users_df=users_df,
        movies_df=movies_df,
        movie_year_map=movie_year_map,
        rate_year_map=rate_year_map,
        user_stats=user_stats,
        movie_stats=movie_stats,
        k=k
    )

    print(f"\n【评估结果】")
    print(f"  HR@{k}:   {hr:.4f}")
    print(f"  NDCG@{k}: {ndcg:.4f}")

    return {'hr': hr, 'ndcg': ndcg}
