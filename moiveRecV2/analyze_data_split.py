"""
数据划分方法对比分析
对比三种数据划分方法:
1. 随机划分
2. 时间划分
3. Leave-One-Out
"""

import pandas as pd
import numpy as np
from collections import Counter
import sys

# 同时输出到文件和控制台
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger('data_split_analysis.txt')

# ========== 1. 加载数据 ==========
print("=" * 80)
print("加载MovieLens 1M数据集...")
print("=" * 80)

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

print(f"总评分记录数: {len(ratings):,}")
print(f"总用户数: {len(users):,}")
print(f"总电影数: {len(movies):,}")
print(f"评分矩阵稀疏度: {len(ratings) / (len(users) * len(movies)) * 100:.2f}%")

# ========== 2. 基础数据分析 ==========
print("" + "=" * 80)
print("基础数据分析")
print("=" * 80)

# 用户评分统计
user_rating_counts = ratings.groupby('UserID').size()
print(f"用户评分次数统计:")
print(f"  最小值: {user_rating_counts.min()}")
print(f"  最大值: {user_rating_counts.max()}")
print(f"  平均值: {user_rating_counts.mean():.2f}")
print(f"  中位数: {user_rating_counts.median():.2f}")
print(f"  标准差: {user_rating_counts.std():.2f}")

# 评分次数分布
print(f"用户评分次数分布:")
print(f"  评分 < 20次的用户: {(user_rating_counts < 20).sum()} ({(user_rating_counts < 20).sum() / len(user_rating_counts) * 100:.2f}%)")
print(f"  评分 20-50次的用户: {((user_rating_counts >= 20) & (user_rating_counts < 50)).sum()} ({((user_rating_counts >= 20) & (user_rating_counts < 50)).sum() / len(user_rating_counts) * 100:.2f}%)")
print(f"  评分 50-100次的用户: {((user_rating_counts >= 50) & (user_rating_counts < 100)).sum()} ({((user_rating_counts >= 50) & (user_rating_counts < 100)).sum() / len(user_rating_counts) * 100:.2f}%)")
print(f"  评分 100-200次的用户: {((user_rating_counts >= 100) & (user_rating_counts < 200)).sum()} ({((user_rating_counts >= 100) & (user_rating_counts < 200)).sum() / len(user_rating_counts) * 100:.2f}%)")
print(f"  评分 >= 200次的用户: {(user_rating_counts >= 200).sum()} ({(user_rating_counts >= 200).sum() / len(user_rating_counts) * 100:.2f}%)")

# 电影评分统计
movie_rating_counts = ratings.groupby('MovieID').size()
print(f"电影被评分次数统计:")
print(f"  最小值: {movie_rating_counts.min()}")
print(f"  最大值: {movie_rating_counts.max()}")
print(f"  平均值: {movie_rating_counts.mean():.2f}")
print(f"  中位数: {movie_rating_counts.median():.2f}")

# 时间跨度分析
ratings['DateTime'] = pd.to_datetime(ratings['Timestamp'], unit='s')
print(f"评分时间跨度:")
print(f"  最早评分: {ratings['DateTime'].min()}")
print(f"  最晚评分: {ratings['DateTime'].max()}")
print(f"  时间跨度: {(ratings['DateTime'].max() - ratings['DateTime'].min()).days} 天")


# ========== 3. 方案1: 随机划分 ==========
print("" + "=" * 80)
print("方案1: 随机划分 (80% train, 10% val, 10% test)")
print("=" * 80)

np.random.seed(42)
shuffled_ratings = ratings.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(shuffled_ratings)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_random = shuffled_ratings[:train_end]
val_random = shuffled_ratings[train_end:val_end]
test_random = shuffled_ratings[val_end:]

print(f"数据集大小:")
print(f"  Train: {len(train_random):,} ({len(train_random)/n*100:.1f}%)")
print(f"  Val:   {len(val_random):,} ({len(val_random)/n*100:.1f}%)")
print(f"  Test:  {len(test_random):,} ({len(test_random)/n*100:.1f}%)")

print(f"用户覆盖率:")
print(f"  Train: {train_random['UserID'].nunique():,} / {len(users):,} ({train_random['UserID'].nunique()/len(users)*100:.2f}%)")
print(f"  Val:   {val_random['UserID'].nunique():,} / {len(users):,} ({val_random['UserID'].nunique()/len(users)*100:.2f}%)")
print(f"  Test:  {test_random['UserID'].nunique():,} / {len(users):,} ({test_random['UserID'].nunique()/len(users)*100:.2f}%)")

print(f"电影覆盖率:")
print(f"  Train: {train_random['MovieID'].nunique():,} / {len(movies):,} ({train_random['MovieID'].nunique()/len(movies)*100:.2f}%)")
print(f"  Val:   {val_random['MovieID'].nunique():,} / {len(movies):,} ({val_random['MovieID'].nunique()/len(movies)*100:.2f}%)")
print(f"  Test:  {test_random['MovieID'].nunique():,} / {len(movies):,} ({test_random['MovieID'].nunique()/len(movies)*100:.2f}%)")

# 冷启动问题分析
train_users = set(train_random['UserID'].unique())
val_users = set(val_random['UserID'].unique())
test_users = set(test_random['UserID'].unique())

print(f"冷启动用户分析:")
print(f"  Val中的新用户(Train中未出现): {len(val_users - train_users)} ({len(val_users - train_users)/len(val_users)*100:.2f}%)")
print(f"  Test中的新用户(Train中未出现): {len(test_users - train_users)} ({len(test_users - train_users)/len(test_users)*100:.2f}%)")

train_movies = set(train_random['MovieID'].unique())
val_movies = set(val_random['MovieID'].unique())
test_movies = set(test_random['MovieID'].unique())

print(f"冷启动电影分析:")
print(f"  Val中的新电影(Train中未出现): {len(val_movies - train_movies)} ({len(val_movies - train_movies)/len(val_movies)*100:.2f}%)")
print(f"  Test中的新电影(Train中未出现): {len(test_movies - train_movies)} ({len(test_movies - train_movies)/len(test_movies)*100:.2f}%)")


# ========== 4. 方案2: 时间划分 ==========
print("" + "=" * 80)
print("方案2: 时间划分 (前80% train, 80%-90% val, 后10% test)")
print("=" * 80)

# 按时间排序
sorted_ratings = ratings.sort_values('Timestamp').reset_index(drop=True)

n = len(sorted_ratings)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_time = sorted_ratings[:train_end]
val_time = sorted_ratings[train_end:val_end]
test_time = sorted_ratings[val_end:]

print(f"数据集大小:")
print(f"  Train: {len(train_time):,} ({len(train_time)/n*100:.1f}%)")
print(f"  Val:   {len(val_time):,} ({len(val_time)/n*100:.1f}%)")
print(f"  Test:  {len(test_time):,} ({len(test_time)/n*100:.1f}%)")

print(f"时间范围:")
print(f"  Train: {train_time['DateTime'].min()} ~ {train_time['DateTime'].max()}")
print(f"  Val:   {val_time['DateTime'].min()} ~ {val_time['DateTime'].max()}")
print(f"  Test:  {test_time['DateTime'].min()} ~ {test_time['DateTime'].max()}")

print(f"用户覆盖率:")
print(f"  Train: {train_time['UserID'].nunique():,} / {len(users):,} ({train_time['UserID'].nunique()/len(users)*100:.2f}%)")
print(f"  Val:   {val_time['UserID'].nunique():,} / {len(users):,} ({val_time['UserID'].nunique()/len(users)*100:.2f}%)")
print(f"  Test:  {test_time['UserID'].nunique():,} / {len(users):,} ({test_time['UserID'].nunique()/len(users)*100:.2f}%)")

print(f"电影覆盖率:")
print(f"  Train: {train_time['MovieID'].nunique():,} / {len(movies):,} ({train_time['MovieID'].nunique()/len(movies)*100:.2f}%)")
print(f"  Val:   {val_time['MovieID'].nunique():,} / {len(movies):,} ({val_time['MovieID'].nunique()/len(movies)*100:.2f}%)")
print(f"  Test:  {test_time['MovieID'].nunique():,} / {len(movies):,} ({test_time['MovieID'].nunique()/len(movies)*100:.2f}%)")

# 冷启动问题分析
train_users_time = set(train_time['UserID'].unique())
val_users_time = set(val_time['UserID'].unique())
test_users_time = set(test_time['UserID'].unique())

print(f"冷启动用户分析:")
print(f"  Val中的新用户(Train中未出现): {len(val_users_time - train_users_time)} ({len(val_users_time - train_users_time)/len(val_users_time)*100:.2f}%)")
print(f"  Test中的新用户(Train中未出现): {len(test_users_time - train_users_time)} ({len(test_users_time - train_users_time)/len(test_users_time)*100:.2f}%)")

train_movies_time = set(train_time['MovieID'].unique())
val_movies_time = set(val_time['MovieID'].unique())
test_movies_time = set(test_time['MovieID'].unique())

print(f"冷启动电影分析:")
print(f"  Val中的新电影(Train中未出现): {len(val_movies_time - train_movies_time)} ({len(val_movies_time - train_movies_time)/len(val_movies_time)*100:.2f}%)")
print(f"  Test中的新电影(Train中未出现): {len(test_movies_time - train_movies_time)} ({len(test_movies_time - train_movies_time)/len(test_movies_time)*100:.2f}%)")

# 分析每个用户在各个集合中的评分次数
print(f"用户评分分布分析:")
user_counts_train = train_time.groupby('UserID').size()
user_counts_val = val_time.groupby('UserID').size()
user_counts_test = test_time.groupby('UserID').size()

# 统计在所有三个集合都出现的用户
users_in_all = set(train_time['UserID'].unique()) & set(val_time['UserID'].unique()) & set(test_time['UserID'].unique())
print(f"  同时出现在Train/Val/Test的用户数: {len(users_in_all)} ({len(users_in_all)/len(users)*100:.2f}%)")

# 对于出现在所有集合的用户,统计他们的评分分布
if len(users_in_all) > 0:
    avg_train = user_counts_train[list(users_in_all)].mean()
    avg_val = user_counts_val[list(users_in_all)].mean()
    avg_test = user_counts_test[list(users_in_all)].mean()
    print(f"  这些用户的平均评分数 - Train: {avg_train:.2f}, Val: {avg_val:.2f}, Test: {avg_test:.2f}")


# ========== 5. 方案3: Leave-One-Out ==========
print("" + "=" * 80)
print("方案3: Leave-One-Out (每用户最后1次→test, 倒数第2次→val, 其余→train)")
print("=" * 80)

# 按用户和时间排序
ratings_sorted_by_user = ratings.sort_values(['UserID', 'Timestamp']).reset_index(drop=True)

train_loo = []
val_loo = []
test_loo = []

for user_id, group in ratings_sorted_by_user.groupby('UserID'):
    n_ratings = len(group)
    
    if n_ratings >= 3:
        # 至少3次评分:最后1次→test,倒数第2次→val,其余→train
        train_loo.append(group.iloc[:-2])
        val_loo.append(group.iloc[-2:-1])
        test_loo.append(group.iloc[-1:])
    elif n_ratings == 2:
        # 只有2次评分:第1次→train,第2次→test,val为空
        train_loo.append(group.iloc[:-1])
        test_loo.append(group.iloc[-1:])
    else:
        # 只有1次评分:放入train
        train_loo.append(group)

train_loo = pd.concat(train_loo, ignore_index=True)
val_loo = pd.concat(val_loo, ignore_index=True) if val_loo else pd.DataFrame()
test_loo = pd.concat(test_loo, ignore_index=True)

print(f"数据集大小:")
print(f"  Train: {len(train_loo):,} ({len(train_loo)/n*100:.1f}%)")
print(f"  Val:   {len(val_loo):,} ({len(val_loo)/n*100:.1f}%)")
print(f"  Test:  {len(test_loo):,} ({len(test_loo)/n*100:.1f}%)")

print(f"用户覆盖率:")
print(f"  Train: {train_loo['UserID'].nunique():,} / {len(users):,} ({train_loo['UserID'].nunique()/len(users)*100:.2f}%)")
print(f"  Val:   {val_loo['UserID'].nunique() if len(val_loo) > 0 else 0:,} / {len(users):,} ({val_loo['UserID'].nunique()/len(users)*100 if len(val_loo) > 0 else 0:.2f}%)")
print(f"  Test:  {test_loo['UserID'].nunique():,} / {len(users):,} ({test_loo['UserID'].nunique()/len(users)*100:.2f}%)")

print(f"电影覆盖率:")
print(f"  Train: {train_loo['MovieID'].nunique():,} / {len(movies):,} ({train_loo['MovieID'].nunique()/len(movies)*100:.2f}%)")
print(f"  Val:   {val_loo['MovieID'].nunique() if len(val_loo) > 0 else 0:,} / {len(movies):,} ({val_loo['MovieID'].nunique()/len(movies)*100 if len(val_loo) > 0 else 0:.2f}%)")
print(f"  Test:  {test_loo['MovieID'].nunique():,} / {len(movies):,} ({test_loo['MovieID'].nunique()/len(movies)*100:.2f}%)")

# 冷启动问题分析
train_users_loo = set(train_loo['UserID'].unique())
val_users_loo = set(val_loo['UserID'].unique()) if len(val_loo) > 0 else set()
test_users_loo = set(test_loo['UserID'].unique())

print(f"\n冷启动用户分析:")
print(f"  Val中的新用户(Train中未出现): {len(val_users_loo - train_users_loo) if len(val_users_loo) > 0 else 0} (0.00%) ← LOO保证无冷启动")
print(f"  Test中的新用户(Train中未出现): {len(test_users_loo - train_users_loo)} (0.00%) ← LOO保证无冷启动")

train_movies_loo = set(train_loo['MovieID'].unique())
val_movies_loo = set(val_loo['MovieID'].unique()) if len(val_loo) > 0 else set()
test_movies_loo = set(test_loo['MovieID'].unique())

print(f"\n冷启动电影分析:")
print(f"  Val中的新电影(Train中未出现): {len(val_movies_loo - train_movies_loo) if len(val_movies_loo) > 0 else 0} ({len(val_movies_loo - train_movies_loo) / len(val_movies_loo) * 100 if len(val_movies_loo) > 0 else 0:.2f}%)")
print(f"  Test中的新电影(Train中未出现): {len(test_movies_loo - train_movies_loo)} ({len(test_movies_loo - train_movies_loo) / len(test_movies_loo) * 100:.2f}%)")

print(f"\n⚠️  LOO方法的电影冷启动问题:")
print(f"  虽然LOO保证了用户无冷启动,但电影冷启动依然存在!")
print(f"  原因: 用户最后评分的电影可能是首次出现的新电影")
print(f"  影响: Test中有 {len(test_movies_loo - train_movies_loo)} 部电影无历史数据")

# 分析用户评分次数
user_rating_dist = user_rating_counts.value_counts().sort_index()
print(f"\n用户评分次数分布:")
for count, num_users in user_rating_dist.head(10).items():
    print(f"  评分{count}次的用户: {num_users}人")
print(f"  ...")


# ========== 6. 综合对比分析 ==========
print("" + "=" * 80)
print("综合对比分析")
print("=" * 80)

comparison_data = {
    '划分方法': ['随机划分', '时间划分', 'Leave-One-Out'],
    'Train大小': [f"{len(train_random):,}", f"{len(train_time):,}", f"{len(train_loo):,}"],
    'Val大小': [f"{len(val_random):,}", f"{len(val_time):,}", f"{len(val_loo):,}"],
    'Test大小': [f"{len(test_random):,}", f"{len(test_time):,}", f"{len(test_loo):,}"],
    'Test用户覆盖': [
        f"{test_random['UserID'].nunique()}",
        f"{test_time['UserID'].nunique()}",
        f"{test_loo['UserID'].nunique()}"
    ],
    'Test冷启动用户': [
        f"{len(test_users - train_users)}",
        f"{len(test_users_time - train_users_time)}",
        "0 ✓"
    ],
    'Test冷启动电影': [
        f"{len(test_movies - train_movies)}",
        f"{len(test_movies_time - train_movies_time)}",
        f"{len(test_movies_loo - train_movies_loo)}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("" + comparison_df.to_string(index=False))

print("" + "=" * 80)
print("优劣势总结")
print("=" * 80)

print("""
【方案1: 随机划分】
✅ 优点:
  1. 实现简单,代码量少
  2. 数据分布均匀,各集合统计特性相似
  3. 几乎无冷启动用户(覆盖率>99%)
  4. 训练集最大(80%),模型能学到更多pattern

❌ 缺点:
  1. 存在"时间泄露"问题 - 用未来数据预测过去
  2. 不符合真实推荐场景(线上是预测未来行为)
  3. 可能高估模型的实际效果
  4. 无法评估模型对时序趋势的捕捉能力

🎯 适用场景: 快速实验、算法原型验证、学术研究(需明确说明)


【方案2: 时间划分】
✅ 优点:
  1. 符合真实推荐场景 - 用历史预测未来
  2. 无时间泄露问题
  3. 能评估模型的时序泛化能力
  4. 更贴近线上A/B测试效果

❌ 缺点:
  1. 可能存在一定冷启动用户(新用户在后期才活跃)
  2. 训练集较小(70%),可能欠拟合
  3. 数据分布可能有偏移(时间早期vs晚期用户行为差异)
  4. 需要分析时间跨度内的数据变化

🎯 适用场景: 工业界应用、线上系统评估、竞赛最终验证


【方案3: Leave-One-Out】
✅ 优点:
  1. 完全消除用户冷启动问题(保证每个test用户在train中出现)
  2. 最大化训练数据利用(>99%)
  3. 评估更稳定(每用户贡献固定数量样本)
  4. 符合推荐场景(预测下一次行为)

❌ 缺点:
  1. Test集较小(每用户仅1条),可能评估不够充分
  2. 无法评估对活跃用户vs非活跃用户的效果差异
  3. 对于评分次数<3的用户无法同时有train/val/test
  4. ⚠️ 电影冷启动问题依然存在甚至可能更严重
     - 用户最后评分的电影可能是新电影
     - Test集中新电影比例可能不低
     - 需要依赖内容特征(Title/Genres)处理

🎯  适用场景: 序列推荐、下一个物品预测、个性化程度要求高的场景


【推荐方案】
🏆 优先推荐: 方案2(时间划分) + 方案3(LOO)组合使用
  - 方案2用于主评估: 衡量真实场景效果
  - 方案3用于辅助分析: 衡量个性化能力、消除冷启动影响
  - 两个指标都好,才说明模型真正robust

📊 具体建议:
  1. 如果追求真实性: 选方案2,并做冷启动专项分析
  2. 如果快速迭代: 先用方案1跑通,再用方案2验证
  3. 如果是竞赛: 看评估规则,通常用方案2或3
  4. 如果研究序列推荐: 必选方案3
""")

print("分析完成!")
