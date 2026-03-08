"""
Wide&Deep模型训练模块 (PyTorch实现 - 联合训练, 隐式反馈版本)
Wide部分: 线性层 (类似逻辑回归)
Deep部分: DNN (深度神经网络)
训练: BCE损失 + 负采样 (每正例采样4个负例)
输出: Sigmoid 概率 (0-1)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class WideAndDeepModel(nn.Module):
    """
    Wide&Deep模型 (PyTorch实现)
    Wide和Deep联合训练,共享输出层
    必须使用 UserID 和 MovieID 的 Embedding
    """
    
    def __init__(self, input_dim, hidden_units=[64, 32],
                 num_users=6041, num_movies=3953,
                 num_ages=7, num_occupations=21, num_movie_years=82, num_rate_years=10,
                 user_emb_dim=32, movie_emb_dim=32, age_emb_dim=8,
                 occupation_emb_dim=16, movie_year_emb_dim=8, rate_year_emb_dim=4,
                 num_stat_features=2, dropout_rate=0.3, wide_l2_reg=0.01):
        """
        初始化Wide&Deep模型 (隐式反馈版本)

        Args:
            input_dim: 输入特征维度 (9列: UserID, MovieID, Gender, Age, Occupation,
                      MoiveYear, RateYear, user_interact_count, movie_interact_count)
            hidden_units: DNN隐藏层单元数列表 (默认[64, 32])
            num_users: 用户总数
            num_movies: 电影总数
            num_ages: 年龄段数量
            num_occupations: 职业数量
            num_movie_years: 电影年份数量
            num_rate_years: 评分年份数量
            user_emb_dim: UserID Embedding维度 (默认32)
            movie_emb_dim: MovieID Embedding维度 (默认32)
            age_emb_dim: Age Embedding维度 (默认8)
            occupation_emb_dim: Occupation Embedding维度 (默认16)
            movie_year_emb_dim: MovieYear Embedding维度 (默认8)
            rate_year_emb_dim: RateYear Embedding维度 (默认4)
            num_stat_features: 统计特征数量 (默认2: user_interact_count + movie_interact_count)
            dropout_rate: Dropout比率 (默认0.3)
            wide_l2_reg: Wide侧L2正则化系数 (默认0.01)
        """
        super(WideAndDeepModel, self).__init__()
        
        self.input_dim    = input_dim
        self.hidden_units = hidden_units
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_ages = num_ages
        self.num_occupations = num_occupations
        self.num_movie_years = num_movie_years
        self.num_rate_years = num_rate_years
        self.num_stat_features = num_stat_features
        
        # Embedding维度
        self.user_emb_dim = user_emb_dim
        self.movie_emb_dim = movie_emb_dim
        self.age_emb_dim = age_emb_dim
        self.occupation_emb_dim = occupation_emb_dim
        self.movie_year_emb_dim = movie_year_emb_dim
        self.rate_year_emb_dim = rate_year_emb_dim
        self.dropout_rate = dropout_rate
        self.wide_l2_reg = wide_l2_reg  # Wide侧L2正则化系数
        
        # Embedding层
        # 注意:为了处理OOV(未见过的ID),每个Embedding层的大小是 max_id + 1
        # num_users和num_movies已经是最大ID值(包含UNK_ID),所以需要+1来容纳索引
        self.user_embedding = nn.Embedding(num_users + 1, user_emb_dim)  # +1: 容纳0到num_users的所有索引
        self.movie_embedding = nn.Embedding(num_movies + 1, movie_emb_dim)
        self.age_embedding = nn.Embedding(num_ages + 1, age_emb_dim)
        self.occupation_embedding = nn.Embedding(num_occupations + 1, occupation_emb_dim)
        self.movie_year_embedding = nn.Embedding(num_movie_years + 1, movie_year_emb_dim)
        self.rate_year_embedding = nn.Embedding(num_rate_years + 1, rate_year_emb_dim)
        
        # Wide部分: 使用 Embedding(num_classes, 1) 实现逻辑回归
        # 数学上等价于 one-hot + 线性权重
        self.wide_user_emb = nn.Embedding(num_users + 1, 1)  # +1: 容纳0到num_users的所有索引
        self.wide_movie_emb = nn.Embedding(num_movies + 1, 1)  # +1: 容纳0到num_movies的所有索引
        self.wide_gender_emb = nn.Embedding(2, 1)  # Gender: 2类 (0/1)
        self.wide_age_emb = nn.Embedding(num_ages + 1, 1)  # Age: 7个年龄段
        self.wide_occupation_emb = nn.Embedding(num_occupations + 1, 1)  # Occupation: 21个职业
        self.wide_movie_year_emb = nn.Embedding(num_movie_years + 1, 1)  # MovieYear: 82个年份
        self.wide_rate_year_emb = nn.Embedding(num_rate_years + 1, 1)  # RateYear: 10个年份
        
        # Wide部分: 统计特征的线性层
        self.wide_stat_linear = nn.Linear(num_stat_features, 1)
        
        # Wide部分: 特征交叉 - Age × MovieYear
        # 计算交叉特征的总数: num_ages × num_movie_years
        num_age_movie_year_cross = (num_ages + 1) * (num_movie_years + 1)
        self.wide_age_movie_year_cross = nn.Embedding(num_age_movie_year_cross, 1)
        
        # Deep部分特征: 所有Embedding + Gender + 统计特征
        # UserID(32) + MovieID(32) + Age(8) + Occupation(16) + MovieYear(8) + RateYear(4) + Gender(1) + 统计特征(6)
        deep_feature_dim = (user_emb_dim + movie_emb_dim + age_emb_dim + 
                           occupation_emb_dim + movie_year_emb_dim + rate_year_emb_dim + 1 + num_stat_features)
        
        print(f"== Wide&Deep 模型特征分配 (隐式反馈版本) ===")
        print(f"Wide部分: 逻辑回归 + 统计特征线性层 + 特征交叉")
        print(f"  - UserID Embedding({num_users}+1, 1)")
        print(f"  - MovieID Embedding({num_movies}+1, 1)")
        print(f"  - Gender/Age/Occupation/MovieYear/RateYear Embedding")
        print(f"  - 统计特征线性层: Linear({num_stat_features}, 1)  [user_interact_count, movie_interact_count]")
        print(f"  - 特征交叉 (Age × MovieYear): Embedding({(num_ages+1)*(num_movie_years+1)}, 1)")
        print(f"Deep部分特征维度: {deep_feature_dim}")
        print(f"  - UserID Embedding: {user_emb_dim}d | MovieID: {movie_emb_dim}d")
        print(f"  - Age: {age_emb_dim}d | Occupation: {occupation_emb_dim}d")
        print(f"  - MovieYear: {movie_year_emb_dim}d | RateYear: {rate_year_emb_dim}d")
        print(f"  - Gender: 1d | 统计特征: {num_stat_features}d | Dropout: {dropout_rate}")
        print(f"输出: Sigmoid 概率 [0, 1]")
        print(f"Wide侧L2正则化: {wide_l2_reg}")
        print(f"=============================")
        
        # Deep部分: 多层神经网络
        deep_layers = []
        prev_dim = deep_feature_dim
        for units in hidden_units:
            deep_layers.append(nn.Linear(prev_dim, units))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        
        # Deep部分的输出层
        deep_layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*deep_layers)
    
    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 (batch_size, 9)
               列索引: [0]UserID, [1]MovieID, [2]Gender, [3]Age, [4]Occupation,
                      [5]MoiveYear, [6]RateYear,
                      [7]user_interact_count, [8]movie_interact_count

        Returns:
            predictions: Sigmoid概率 (batch_size,) - 范围 [0, 1]
        """
        # 提取各个特征 (注意转为long类型用于Embedding索引)
        user_ids       = x[:, 0].long()
        movie_ids      = x[:, 1].long()
        gender_ids     = x[:, 2].long()  # Gender作为类别索引
        gender         = x[:, 2:3]       # 保持2D形状用于Deep部分 (batch_size, 1)
        age_ids        = x[:, 3].long()
        occupation_ids = x[:, 4].long()
        movie_year_ids = x[:, 5].long()
        rate_year_ids  = x[:, 6].long()

        # 统计特征 (数值特征: user_interact_count, movie_interact_count)
        stat_features = x[:, 7:9]  # (batch_size, 2)
        
        # Deep部分: 获取所有Embedding
        user_emb = self.user_embedding(user_ids)  # (batch_size, user_emb_dim)
        movie_emb = self.movie_embedding(movie_ids)  # (batch_size, movie_emb_dim)
        age_emb = self.age_embedding(age_ids)  # (batch_size, age_emb_dim)
        occupation_emb = self.occupation_embedding(occupation_ids)  # (batch_size, occupation_emb_dim)
        movie_year_emb = self.movie_year_embedding(movie_year_ids)  # (batch_size, movie_year_emb_dim)
        rate_year_emb = self.rate_year_embedding(rate_year_ids)  # (batch_size, rate_year_emb_dim)
        
        # Wide部分: 逻辑回归 + 统计特征线性层 + 特征交叉
        # 计算 Age × MovieYear 的交叉特征索引
        # 公式: cross_id = age_id * (num_movie_years + 1) + movie_year_id
        age_movie_year_cross_ids = age_ids * (self.num_movie_years + 1) + movie_year_ids
        
        wide_out = (
            self.wide_user_emb(user_ids).squeeze(-1) +
            self.wide_movie_emb(movie_ids).squeeze(-1) +
            self.wide_gender_emb(gender_ids).squeeze(-1) +
            self.wide_age_emb(age_ids).squeeze(-1) +
            self.wide_occupation_emb(occupation_ids).squeeze(-1) +
            self.wide_movie_year_emb(movie_year_ids).squeeze(-1) +
            self.wide_rate_year_emb(rate_year_ids).squeeze(-1) +
            self.wide_stat_linear(stat_features).squeeze(-1) +  # 统计特征的线性变换
            self.wide_age_movie_year_cross(age_movie_year_cross_ids).squeeze(-1)  # Age × MovieYear 交叉特征
        ).unsqueeze(-1)  # (batch_size, 1)
        
        # Deep部分: 拼接所有Embedding + Gender + 统计特征
        deep_input = torch.cat([
            user_emb, movie_emb, age_emb, occupation_emb, 
            movie_year_emb, rate_year_emb, gender, stat_features
        ], dim=1)
        
        # Deep部分输出
        deep_out = self.deep(deep_input)
        
        # 联合输出 (Wide和Deep权重平衡)
        output = wide_out * 0.5 + deep_out * 0.5

        # Sigmoid: 将输出转换为 [0, 1] 概率
        return torch.sigmoid(output.squeeze())
    
    def get_wide_l2_loss(self):
        """
        计算Wide侧所有参数的L2正则化损失
        
        Returns:
            torch.Tensor: L2正则化损失值
        """
        l2_loss = 0.0
        
        # Wide侧的所有Embedding层参数
        wide_params = [
            self.wide_user_emb.weight,
            self.wide_movie_emb.weight,
            self.wide_gender_emb.weight,
            self.wide_age_emb.weight,
            self.wide_occupation_emb.weight,
            self.wide_movie_year_emb.weight,
            self.wide_rate_year_emb.weight,
            self.wide_age_movie_year_cross.weight,
        ]
        
        # 统计特征线性层的参数
        wide_params.append(self.wide_stat_linear.weight)
        if self.wide_stat_linear.bias is not None:
            wide_params.append(self.wide_stat_linear.bias)
        
        # 计算L2范数
        for param in wide_params:
            l2_loss += torch.sum(param ** 2)
        
        return self.wide_l2_reg * l2_loss
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=10, batch_size=512, learning_rate=0.001, verbose=True,
            early_stopping_patience=5, early_stopping_min_delta=0.0001,
            neg_sample_ratio=4, all_movie_ids=None):
        """
        训练Wide&Deep模型 (隐式反馈 + 负采样 + BCE损失)

        Args:
            X_train: 训练特征 (numpy array, 全为正例)
            y_train: 训练标签 (全为1, 占位用)
            X_val: 验证特征 (可选, 全为正例)
            y_val: 验证标签 (可选)
            epochs: 训练轮数
            batch_size: 批次大小 (正例batch大小, 实际batch含负例)
            learning_rate: 学习率
            verbose: 是否显示训练详情
            early_stopping_patience: Early Stopping耐心值 (默认5, 设为0禁用)
            early_stopping_min_delta: Early Stopping最小改善阈值 (默认0.0001)
            neg_sample_ratio: 每个正例采样的负例数量 (默认4)
            all_movie_ids: 所有电影ID列表 (用于负采样), 若为None则从训练集提取
        """
        print("训练Wide&Deep模型 (隐式反馈 + 负采样 + BCE)...")
        print(f"  损失函数: Binary Cross Entropy (BCE)")
        print(f"  负采样比例: 1 正 : {neg_sample_ratio} 负")
        print(f"  使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        if early_stopping_patience > 0 and X_val is not None:
            print(f"  Early Stopping: 启用 (patience={early_stopping_patience}, "
                  f"min_delta={early_stopping_min_delta})")
        else:
            print(f"  Early Stopping: 禁用")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # 构建负采样所需的数据结构
        # X_train 中: [0]=UserID, [1]=MovieID
        train_user_ids  = X_train[:, 0].astype(int)
        train_movie_ids = X_train[:, 1].astype(int)

        # 每个用户已交互的电影集合
        user_pos_set = {}
        for uid, mid in zip(train_user_ids, train_movie_ids):
            if uid not in user_pos_set:
                user_pos_set[uid] = set()
            user_pos_set[uid].add(mid)

        # 候选电影池
        if all_movie_ids is None:
            all_movie_arr = np.array(sorted(set(train_movie_ids)))
        else:
            all_movie_arr = np.array(sorted(all_movie_ids))

        # 正例 DataLoader
        X_train_tensor = torch.FloatTensor(X_train)
        train_dataset  = TensorDataset(X_train_tensor, torch.arange(len(X_train)))
        train_loader   = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 验证集 (正例 + 负例各一批, 用于监控 BCE)
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.ones(len(X_val), dtype=torch.float32)
            val_dataset  = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        best_val_loss    = float('inf')
        patience_counter = 0
        best_model_state = None
        rng = np.random.default_rng(0)

        for epoch in range(epochs):
            self.train()
            train_bce_loss = 0.0
            train_total    = 0

            for batch_X, batch_idx in train_loader:
                batch_idx_np = batch_idx.numpy()
                batch_X_np   = batch_X.numpy()

                # --- 动态负采样 ---
                neg_rows = []
                for i, idx in enumerate(batch_idx_np):
                    uid    = int(batch_X_np[i, 0])
                    pos_set = user_pos_set.get(uid, set())
                    # 从候选池中随机采样, 过滤已交互
                    candidates = all_movie_arr[~np.isin(all_movie_arr, list(pos_set))]
                    if len(candidates) == 0:
                        candidates = all_movie_arr
                    sampled = rng.choice(candidates,
                                         size=neg_sample_ratio,
                                         replace=len(candidates) < neg_sample_ratio)
                    for neg_mid in sampled:
                        neg_row        = batch_X_np[i].copy()
                        neg_row[1]     = neg_mid   # 替换 MovieID
                        neg_rows.append(neg_row)

                neg_X  = torch.FloatTensor(np.array(neg_rows))
                neg_y  = torch.zeros(len(neg_rows), dtype=torch.float32)
                pos_y  = torch.ones(len(batch_X),   dtype=torch.float32)

                # 拼接正负例
                combined_X = torch.cat([batch_X, neg_X], dim=0).to(device)
                combined_y = torch.cat([pos_y,   neg_y], dim=0).to(device)

                optimizer.zero_grad()
                outputs  = self(combined_X)
                bce_loss = criterion(outputs, combined_y)
                l2_loss  = self.get_wide_l2_loss()
                loss     = bce_loss + l2_loss

                loss.backward()
                optimizer.step()

                train_bce_loss += bce_loss.item() * len(combined_X)
                train_total    += len(combined_X)

            train_avg_loss = train_bce_loss / train_total

            # 验证
            if val_loader is not None:
                self.eval()
                val_bce_loss = 0.0
                val_total    = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs  = self(batch_X)
                        loss     = criterion(outputs, batch_y)
                        val_bce_loss += loss.item() * len(batch_X)
                        val_total    += len(batch_X)

                val_avg_loss = val_bce_loss / val_total

                if verbose:
                    print(f"  Epoch [{epoch+1}/{epochs}] "
                          f"Train BCE: {train_avg_loss:.4f} | "
                          f"Val BCE: {val_avg_loss:.4f}", end="")

                if early_stopping_patience > 0:
                    if val_avg_loss < best_val_loss - early_stopping_min_delta:
                        best_val_loss    = val_avg_loss
                        patience_counter = 0
                        best_model_state = {k: v.clone() for k, v in self.state_dict().items()}
                        if verbose:
                            print(" ✓ [Best]")
                    else:
                        patience_counter += 1
                        if verbose:
                            print(f" (patience: {patience_counter}/{early_stopping_patience})")
                        if patience_counter >= early_stopping_patience:
                            print(f"  Early Stopping触发! 最佳Val BCE: {best_val_loss:.4f}")
                            self.load_state_dict(best_model_state)
                            break
                else:
                    if verbose:
                        print()
            else:
                if verbose:
                    print(f"  Epoch [{epoch+1}/{epochs}] Train BCE: {train_avg_loss:.4f}")

        if early_stopping_patience > 0 and val_loader is not None and best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"  训练完成,使用最佳模型 (Val BCE: {best_val_loss:.4f})")

        print("Wide&Deep模型训练完成")
    
    def predict(self, X):
        """
        预测评分
        
        Args:
            X: 特征矩阵 (numpy array)
        
        Returns:
            array: 预测评分 (n_samples,)
        """
        self.eval()
        device = next(self.parameters()).device
        
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            predictions = self(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save(self, filepath='models/wide_deep_model.pth'):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型状态和配置
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_units': self.hidden_units,
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            'num_ages': self.num_ages,
            'num_occupations': self.num_occupations,
            'num_movie_years': self.num_movie_years,
            'num_rate_years': self.num_rate_years,
            'num_stat_features': self.num_stat_features,
            'user_emb_dim': self.user_emb_dim,
            'movie_emb_dim': self.movie_emb_dim,
            'age_emb_dim': self.age_emb_dim,
            'occupation_emb_dim': self.occupation_emb_dim,
            'movie_year_emb_dim': self.movie_year_emb_dim,
            'rate_year_emb_dim': self.rate_year_emb_dim
        }
        
        torch.save(checkpoint, filepath)
        print(f"  Wide&Deep模型已保存至: {filepath}")
    
    @staticmethod
    def load(filepath='models/wide_deep_model.pth', hidden_units=[64, 32],
             num_users=6041, num_movies=3953,
             num_ages=7, num_occupations=21, num_movie_years=82, num_rate_years=10,
             user_emb_dim=32, movie_emb_dim=32, age_emb_dim=8,
             occupation_emb_dim=16, movie_year_emb_dim=8, rate_year_emb_dim=4,
             num_stat_features=2, dropout_rate=0.3, wide_l2_reg=0.01):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            hidden_units: DNN隐藏层单元数 (需要与训练时一致,默认[64, 32])
            num_users: 用户总数
            num_movies: 电影总数
            num_ages: 年龄段数量
            num_occupations: 职业数量
            num_movie_years: 电影年份数量
            num_rate_years: 评分年份数量
            user_emb_dim: UserID Embedding维度
            movie_emb_dim: MovieID Embedding维度
            age_emb_dim: Age Embedding维度
            occupation_emb_dim: Occupation Embedding维度
            movie_year_emb_dim: MovieYear Embedding维度
            rate_year_emb_dim: RateYear Embedding维度
            num_stat_features: 统计特征数量
            dropout_rate: Dropout比率
            wide_l2_reg: Wide侧L2正则化系数
        
        Returns:
            WideAndDeepModel: 加载的模型,如果文件不存在则返回None
        """
        if not os.path.exists(filepath):
            return None
        
        # 加载checkpoint (PyTorch 2.6+ 需要设置 weights_only=False)
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # 创建模型实例 (优先使用保存的配置)
        model = WideAndDeepModel(
            input_dim=checkpoint['input_dim'],
            hidden_units=checkpoint.get('hidden_units', hidden_units),
            num_users=checkpoint.get('num_users', num_users),
            num_movies=checkpoint.get('num_movies', num_movies),
            num_ages=checkpoint.get('num_ages', num_ages),
            num_occupations=checkpoint.get('num_occupations', num_occupations),
            num_movie_years=checkpoint.get('num_movie_years', num_movie_years),
            num_rate_years=checkpoint.get('num_rate_years', num_rate_years),
            user_emb_dim=checkpoint.get('user_emb_dim', user_emb_dim),
            movie_emb_dim=checkpoint.get('movie_emb_dim', movie_emb_dim),
            age_emb_dim=checkpoint.get('age_emb_dim', age_emb_dim),
            occupation_emb_dim=checkpoint.get('occupation_emb_dim', occupation_emb_dim),
            movie_year_emb_dim=checkpoint.get('movie_year_emb_dim', movie_year_emb_dim),
            rate_year_emb_dim=checkpoint.get('rate_year_emb_dim', rate_year_emb_dim),
            num_stat_features=checkpoint.get('num_stat_features', num_stat_features),
            dropout_rate=dropout_rate,
            wide_l2_reg=wide_l2_reg
        )
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"  已加载保存的Wide&Deep模型: {filepath}")
        return model


def train_wide_deep_model(X_train, y_train, X_val=None, y_val=None,
                          hidden_units=[64, 32], epochs=10, batch_size=512,
                          learning_rate=0.001, verbose=True,
                          num_users=6041, num_movies=3953,
                          num_ages=7, num_occupations=21,
                          num_movie_years=82, num_rate_years=10,
                          num_stat_features=2, dropout_rate=0.3, wide_l2_reg=0.01,
                          early_stopping_patience=5, early_stopping_min_delta=0.0001,
                          neg_sample_ratio=4, all_movie_ids=None):
    """
    训练Wide&Deep模型 (隐式反馈 + 负采样 + BCE)

    Args:
        X_train: 训练特征 (9列: UserID, MovieID, Gender, Age, Occupation, MoiveYear, RateYear,
                user_interact_count, movie_interact_count), 全为正例
        y_train: 训练标签 (全为1, 负例在训练中动态采样)
        X_val: 验证特征 (可选)
        y_val: 验证标签 (可选)
        hidden_units: DNN隐藏层单元数 (默认[64, 32])
        epochs: 训练轮数
        batch_size: 批次大小 (正例batch大小)
        learning_rate: 学习率
        verbose: 是否显示训练详情
        num_users: 用户总数
        num_movies: 电影总数
        num_ages: 年龄段数量
        num_occupations: 职业数量
        num_movie_years: 电影年份数量
        num_rate_years: 评分年份数量
        num_stat_features: 统计特征数量 (默认2)
        dropout_rate: Dropout比率
        wide_l2_reg: Wide侧L2正则化系数
        early_stopping_patience: Early Stopping耐心值
        early_stopping_min_delta: Early Stopping最小改善阈值
        neg_sample_ratio: 每正例负例数 (默认4)
        all_movie_ids: 全量电影ID列表 (用于负采样)

    Returns:
        WideAndDeepModel: 训练好的模型
    """
    print("\n[4/5] 训练Wide&Deep模型 (隐式反馈)...")
    print(f"  Wide部分: 逻辑回归 + 统计特征线性层")
    print(f"    - 统计特征: {num_stat_features}维 (user_interact_count + movie_interact_count)")
    print(f"  Deep部分: DNN {hidden_units}")
    print(f"  联合训练: Wide和Deep权重平衡 (0.5 + 0.5)")
    print(f"  损失函数: BCE | 负采样比: 1:{neg_sample_ratio}")

    input_dim = X_train.shape[1]
    model = WideAndDeepModel(
        input_dim,
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

    model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=verbose,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        neg_sample_ratio=neg_sample_ratio,
        all_movie_ids=all_movie_ids
    )

    return model


def predict(model, X):
    """
    使用Wide&Deep模型进行预测

    Args:
        model: 训练好的Wide&Deep模型
        X: 特征矩阵

    Returns:
        array: 预测交互概率 (n_samples,), 范围 [0, 1]
    """
    return model.predict(X)
