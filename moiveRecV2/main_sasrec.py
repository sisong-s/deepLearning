"""
MovieLens 1M - SASRec 序列推荐版本
数据集划分: 基于用户的 Leave-Last-2-Out (train / val / test)
训练:       SASRec (Self-Attentive Sequential Recommendation)
            隐式反馈 + BCE + 动态负采样 (1:1)
评估:       HR@10, NDCG@10 (每用户 100 个随机负例)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from data_pipeline import load_data, split_data_by_user
from feature_engineering import preprocess_users, preprocess_movies, preprocess_ratings
from model_sasrec import (
    SASRecModel, SASRecPredictor, train_sasrec_model, _pad_sequence
)
from evaluation import evaluate_model


# ──────────────────────────────────────────
# 辅助: 从 ratings 构建用户历史序列
# ──────────────────────────────────────────

def build_user_sequences(ratings_df):
    """
    将评分记录按用户 + 时间排序, 构建每个用户的交互序列.

    Args:
        ratings_df: DataFrame with columns [UserID, MovieID, Timestamp]

    Returns:
        dict {user_id -> list[movie_id]}  — 按时间升序排列
    """
    sorted_df = ratings_df.sort_values(['UserID', 'Timestamp'])
    user_seqs = (
        sorted_df.groupby('UserID')['MovieID']
        .apply(list)
        .to_dict()
    )
    return user_seqs


# ──────────────────────────────────────────
# 评估适配器: 将 SASRec 包装成与 evaluation.py 兼容的接口
# ──────────────────────────────────────────

class _SASRecEvalWrapper:
    """
    供 evaluation.py 的 calculate_hr_and_ndcg_at_k 调用.
    接受与 WideAndDeep 相同的 predict(X) 接口:
        X: (N, 9) 但 SASRec 只用前 2 列 [UserID, MovieID]
    """

    def __init__(self, sasrec_predictor: SASRecPredictor):
        self._pred = sasrec_predictor

    def predict(self, X):
        """X: (N, ≥2) numpy array"""
        return self._pred.predict(X)


# ──────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────

def main():
    print("=" * 80)
    print("MovieLens 1M - SASRec 序列推荐 (隐式反馈 + BCE + 负采样)")
    print("=" * 80)

    # ── 1. 加载数据 ──────────────────────────────
    ratings, users, movies = load_data()

    # ── 2. 按用户划分数据集 (Leave-Last-2-Out) ───
    train_data, val_data, test_data = split_data_by_user(ratings)

    # ── 3. 构建用户交互序列 ───────────────────────
    print("\n[3/5] 构建用户交互序列...")

    # 训练序列 (用于模型输入)
    user_train_seqs = build_user_sequences(train_data)
    # 验证正例: 每用户 val_data 中的电影 (倒数第 2 条)
    user_val_pos    = dict(zip(val_data['UserID'], val_data['MovieID']))
    # 测试正例: 每用户 test_data 中的电影 (最后 1 条)
    user_test_pos   = dict(zip(test_data['UserID'], test_data['MovieID']))

    # 全量电影 ID (用于负采样)
    all_movie_ids = list(ratings['MovieID'].unique())
    num_items     = int(ratings['MovieID'].max())  # 最大 ID 作为 num_items

    print(f"  训练序列用户数: {len(user_train_seqs)}")
    seq_lengths = [len(s) for s in user_train_seqs.values()]
    print(f"  序列长度 — 均值: {np.mean(seq_lengths):.1f}, "
          f"中位数: {np.median(seq_lengths):.1f}, "
          f"最大: {max(seq_lengths)}, 最小: {min(seq_lengths)}")
    print(f"  验证集用户数: {len(user_val_pos)}")
    print(f"  测试集用户数: {len(user_test_pos)}")
    print(f"  全量电影数:   {len(all_movie_ids)}")
    print(f"  num_items (max_id): {num_items}")

    # ── 4. 模型配置 ───────────────────────────────
    model_path        = 'models/sasrec_model.pth'
    hidden_dim        = 32
    max_seq_len       = 50
    num_blocks        = 2
    num_heads         = 2
    dropout_rate      = 0.2
    item_emb_l2       = 1e-6
    epochs            = 30
    batch_size        = 2048
    learning_rate     = 1e-3
    neg_sample_ratio  = 1
    early_stopping_patience   = 3
    early_stopping_min_delta  = 1e-4

    print(f"\n模型参数配置:")
    print(f"  num_items={num_items}, hidden_dim={hidden_dim}, max_seq_len={max_seq_len}")
    print(f"  num_blocks={num_blocks}, num_heads={num_heads}, dropout={dropout_rate}")

    # ── 尝试加载已有模型 ──────────────────────────
    sasrec_model = SASRecModel.load(
        model_path,
        num_items=num_items,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        item_emb_l2=item_emb_l2,
    )

    if sasrec_model is None:
        sasrec_model = train_sasrec_model(
            user_train_seqs=user_train_seqs,
            all_movie_ids=all_movie_ids,
            user_val_pos=user_val_pos,
            num_items=num_items,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            item_emb_l2=item_emb_l2,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            neg_sample_ratio=neg_sample_ratio,
            verbose=True,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )
        sasrec_model.save(model_path)
    else:
        print(f"[4/5] 使用已保存的模型: {model_path}")

    # ── 5. 评估 ───────────────────────────────────
    print("\n[5/5] 评估 SASRec 模型 (HR@10, NDCG@10)...")
    print(f"  评估方式: Leave-Last-1-Out + 100 随机负例")
    print(f"  测试用户数: {len(user_test_pos)}")

    # 构建测试负例候选池 (每用户 100 个负例, 排除所有已交互电影)
    all_interactions = (
        pd.concat([train_data[['UserID', 'MovieID']],
                   val_data[['UserID', 'MovieID']],
                   test_data[['UserID', 'MovieID']]])
        .groupby('UserID')['MovieID']
        .apply(set)
        .to_dict()
    )

    rng = np.random.default_rng(42)
    all_movie_arr   = np.array(sorted(all_movie_ids))
    test_neg_samples = {}
    for uid, pos_mid in user_test_pos.items():
        interacted = all_interactions.get(uid, set())
        candidates = [m for m in all_movie_arr if m not in interacted]
        if len(candidates) >= 100:
            neg_movies = rng.choice(candidates, size=100, replace=False).tolist()
        else:
            neg_movies = candidates
        test_neg_samples[uid] = neg_movies

    print(f"  已构建 {len(test_neg_samples)} 个用户的负例候选池")

    # ── 构建 SASRec 专用评估 (HR@10 / NDCG@10) ──
    hr_list   = []
    ndcg_list = []

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sasrec_model.to(device)
    sasrec_model.eval()

    with torch.no_grad():
        for uid, pos_mid in user_test_pos.items():
            neg_movies = test_neg_samples.get(uid, [])
            if len(neg_movies) == 0:
                continue

            candidates = [pos_mid] + list(neg_movies)  # 101 个候选
            seq        = user_train_seqs.get(uid, [])
            padded_seq = _pad_sequence(seq, max_seq_len)

            seq_t   = torch.LongTensor([padded_seq] * len(candidates)).to(device)  # (101, L)
            item_t  = torch.LongTensor(candidates).to(device)                       # (101,)
            scores  = sasrec_model(seq_t, item_t).cpu().numpy()                     # (101,)

            ranked  = np.argsort(-scores)
            top10   = ranked[:10]

            hit  = 1 if 0 in top10 else 0
            hr_list.append(hit)

            if hit:
                rank = int(np.where(ranked == 0)[0][0]) + 1
                ndcg = 1.0 / np.log2(rank + 1)
            else:
                ndcg = 0.0
            ndcg_list.append(ndcg)

    hr   = float(np.mean(hr_list))   if hr_list   else 0.0
    ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0

    print(f"\n【评估结果】")
    print(f"  HR@10:   {hr:.4f}")
    print(f"  NDCG@10: {ndcg:.4f}")

    print("\n" + "=" * 80)
    print("实验完成!")
    print(f"  HR@10:   {hr:.4f}")
    print(f"  NDCG@10: {ndcg:.4f}")
    print("=" * 80)

    return {'hr': hr, 'ndcg': ndcg}


if __name__ == "__main__":
    main()
