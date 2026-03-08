"""
SASRec (Self-Attentive Sequential Recommendation) 模型训练模块
论文: Self-Attentive Sequential Recommendation (Wang-Cheng Kang, Julian McAuley, ICDM 2018)

核心思想:
  - 用 Transformer 自注意力机制对用户的交互序列建模
  - 捕捉用户的长短期兴趣
  - 预测: 用序列上下文向量与候选电影 Embedding 做内积打分 → Sigmoid 概率

训练:
  - 隐式反馈 (正例=实际交互, 负例=随机采样)
  - BCE 损失 + 动态负采样 (1:1)
  - Early Stopping

评估:
  - HR@10, NDCG@10 (每用户 100 个随机负例)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ──────────────────────────────────────────
# 1. SASRec 核心模块
# ──────────────────────────────────────────

class PointWiseFeedForward(nn.Module):
    """逐位置前馈网络 (Position-wise FFN)"""

    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.ln(x + residual)


class SASRecBlock(nn.Module):
    """单个 Transformer Block: 多头自注意力 + FFN + 残差 + LayerNorm"""

    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.ffn = PointWiseFeedForward(hidden_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            attn_mask: 因果掩码 (seq_len, seq_len), 防止未来信息泄露
            key_padding_mask: padding掩码 (batch, seq_len), True=需要mask的位置
        """
        residual = x
        x_norm = self.ln_attn(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = residual + self.dropout(attn_out)
        x = self.ffn(x)
        return x


# ──────────────────────────────────────────
# 2. SASRec 主模型
# ──────────────────────────────────────────

class SASRecModel(nn.Module):
    """
    SASRec 序列推荐模型

    输入:
        item_seq: 用户历史交互序列 (batch, max_seq_len) — MovieID 序列, 0=padding
        target_items: 候选电影 ID (batch,)

    输出:
        Sigmoid 概率 (batch,) ∈ [0, 1]

    模型结构:
        Item Embedding → Positional Embedding → N × SASRecBlock → 取最后有效位置
        → 与 target item Embedding 内积 → Sigmoid
    """

    def __init__(self, num_items, hidden_dim=64, max_seq_len=50,
                 num_blocks=2, num_heads=2, dropout_rate=0.2,
                 item_emb_l2=0.0):
        """
        Args:
            num_items: 物品总数 (含 padding=0 占位, 实际 ID 从 1 开始)
            hidden_dim: Embedding / 隐层维度 (默认 64)
            max_seq_len: 最大序列长度 (默认 50)
            num_blocks: Transformer Block 数量 (默认 2)
            num_heads: 多头注意力头数 (默认 2)
            dropout_rate: Dropout 比率 (默认 0.2)
            item_emb_l2: Item Embedding L2 正则化系数 (默认 0)
        """
        super().__init__()

        self.num_items    = num_items
        self.hidden_dim   = hidden_dim
        self.max_seq_len  = max_seq_len
        self.num_blocks   = num_blocks
        self.num_heads    = num_heads
        self.dropout_rate = dropout_rate
        self.item_emb_l2  = item_emb_l2

        # Item Embedding: 0 为 padding, 索引范围 [0, num_items]
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        # 位置 Embedding: 固定最大长度
        self.pos_emb  = nn.Embedding(max_seq_len + 1, hidden_dim)

        self.emb_dropout = nn.Dropout(dropout_rate)
        self.ln_emb      = nn.LayerNorm(hidden_dim)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])

        self.ln_out = nn.LayerNorm(hidden_dim)

        self._init_weights()

        print(f"== SASRec 模型配置 ==")
        print(f"  num_items={num_items}, hidden_dim={hidden_dim}")
        print(f"  max_seq_len={max_seq_len}, num_blocks={num_blocks}, num_heads={num_heads}")
        print(f"  dropout_rate={dropout_rate}, item_emb_l2={item_emb_l2}")
        print(f"====================")

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight,  std=0.01)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.xavier_uniform_(block.ffn.fc1.weight)
            nn.init.xavier_uniform_(block.ffn.fc2.weight)

    def _encode_sequence(self, item_seq):
        """
        将 item 序列编码为上下文向量 (取最后有效位置)

        Args:
            item_seq: (batch, seq_len) — 0 为 padding

        Returns:
            seq_output: (batch, hidden_dim)
        """
        batch_size, seq_len = item_seq.shape
        device = item_seq.device

        # ── Item + Position Embedding ──
        positions = torch.arange(1, seq_len + 1, device=device).unsqueeze(0)  # (1, seq_len)
        positions = positions.expand(batch_size, -1)
        # padding 位置的 position 不影响注意力 (后面用 key_padding_mask 屏蔽)                 # (batch, seq_len)
        x = self.item_emb(item_seq) + self.pos_emb(positions)                 # (batch, seq_len, d)
        x = self.ln_emb(self.emb_dropout(x))

        # ── 因果掩码 (上三角 = -inf 防止看到未来) ──
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        ).to(x.dtype)

        # ── padding 位置掩码: True = 该位置是 padding ──
        pad_mask = (item_seq == 0)   # (batch, seq_len)

        # ── 不使用 key_padding_mask, 改为 attention 后对 padding 位置输出归零 ──
        # 原因: 因果掩码 + key_padding_mask 联合使用时, padding query 能看到的 key
        #       全被屏蔽 (causal 只允许看左侧, 左侧全是 padding 又被 key_padding_mask 屏蔽)
        #       → softmax(全-inf) = NaN, 导致后续计算崩溃
        # 解决: padding token 照常参与 attention key/value 计算, 仅在输出后清零其向量,
        #       不让 padding 位置的噪声向量影响序列表示
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=None)
            # 将 padding 位置的输出归零, 避免残差累积引入噪声
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        x = self.ln_out(x)   # (batch, seq_len, d)

        # ── 取每条序列最后一个非 padding 位置的向量 ──
        seq_lengths = (item_seq != 0).sum(dim=1).clamp(min=1)  # (batch,) 全padding时取位置1
        last_idx    = (seq_lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.hidden_dim)
        seq_output  = x.gather(1, last_idx).squeeze(1)  # (batch, hidden_dim)

        return seq_output

    def forward(self, item_seq, target_items):
        """
        前向传播

        Args:
            item_seq:     (batch, max_seq_len) — 用户历史序列
            target_items: (batch,) — 候选物品 ID

        Returns:
            probs: (batch,) Sigmoid 概率
        """
        seq_vec    = self._encode_sequence(item_seq)          # (batch, d)
        target_emb = self.item_emb(target_items)              # (batch, d)
        logits     = (seq_vec * target_emb).sum(dim=-1)       # (batch,) 内积
        # clamp 防止 NaN/Inf 传入 sigmoid (全 padding 序列的极端情况)
        logits     = torch.clamp(logits, min=-30.0, max=30.0)
        return torch.sigmoid(logits)

    def get_l2_loss(self):
        """计算 Item Embedding L2 正则化损失"""
        if self.item_emb_l2 <= 0:
            return 0.0
        return self.item_emb_l2 * torch.sum(self.item_emb.weight ** 2)

    def predict_scores(self, item_seq, target_items):
        """
        预测一批候选物品的分数 (eval 模式下调用)

        Args:
            item_seq:     (batch, max_seq_len) — 用户历史序列
            target_items: (batch, num_candidates) — 候选物品 ID 列表

        Returns:
            scores: (batch, num_candidates)
        """
        seq_vec    = self._encode_sequence(item_seq)   # (batch, d)
        target_emb = self.item_emb(target_items)       # (batch, num_candidates, d)
        scores     = (seq_vec.unsqueeze(1) * target_emb).sum(dim=-1)  # (batch, num_candidates)
        scores     = torch.clamp(scores, min=-30.0, max=30.0)
        return scores

    # ──────────────────────────────────────────
    # 3. Dataset
    # ──────────────────────────────────────────

    def fit(self, user_train_seqs, all_movie_ids,
            user_val_pos=None,
            epochs=30, batch_size=256, learning_rate=1e-3,
            neg_sample_ratio=1, verbose=True,
            early_stopping_patience=5, early_stopping_min_delta=1e-4):
        """
        训练 SASRec 模型 (BCE + 动态负采样)

        Args:
            user_train_seqs: dict {user_id -> list[movie_id]} — 按时间排序的训练交互序列
            all_movie_ids:   list — 全量电影 ID (负采样候选池)
            user_val_pos:    dict {user_id -> movie_id} — 验证集每用户的正例 (用于 Val BCE)
            epochs:          训练轮数
            batch_size:      批大小
            learning_rate:   Adam 学习率
            neg_sample_ratio: 每正例采样的负例数 (默认 1)
            verbose:         打印训练进度
            early_stopping_patience: Early Stopping 耐心值 (基于 Val BCE, 设 0 禁用)
            early_stopping_min_delta: Early Stopping 最小改善阈值
        """
        print("训练 SASRec 模型 (隐式反馈 + BCE + 负采样)...")
        print(f"  损失函数: BCE | 负采样比: 1:{neg_sample_ratio}")
        print(f"  设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        if early_stopping_patience > 0 and user_val_pos:
            print(f"  Early Stopping: patience={early_stopping_patience}, "
                  f"min_delta={early_stopping_min_delta} (基于 Val BCE)")
        else:
            print("  Early Stopping: 禁用")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        all_movie_arr = np.array(sorted(all_movie_ids))
        # 每个用户已交互的电影集合 (用于负采样过滤)
        user_pos_set  = {uid: set(seq) for uid, seq in user_train_seqs.items()}

        # 构建训练样本列表: [(user_id, seq_tensor, pos_movie_id), ...]
        train_samples = _build_train_samples(user_train_seqs, self.max_seq_len)

        # 构建验证样本: 每个 val 正例 + 1 个随机负例
        val_samples = None
        if user_val_pos:
            val_samples = _build_val_samples(
                user_val_pos, user_train_seqs, user_pos_set,
                all_movie_arr, self.max_seq_len
            )

        dataset = _SASRecTrainDataset(train_samples)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=False)

        criterion  = nn.BCELoss()
        optimizer  = optim.Adam(self.parameters(), lr=learning_rate)

        best_val_loss    = float('inf')
        patience_counter = 0
        best_state       = None
        rng              = np.random.default_rng(42)

        epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch",
                         dynamic_ncols=True, colour="green")

        for epoch in epoch_bar:
            # ── 训练 ──────────────────────────────────
            self.train()
            total_loss = 0.0
            total_cnt  = 0

            batch_bar = tqdm(loader, desc=f"  Epoch {epoch+1:>3}/{epochs}",
                             unit="batch", leave=False, dynamic_ncols=True)

            for batch_uids, batch_seqs, batch_pos in batch_bar:
                batch_uids = batch_uids.numpy()


                batch_seqs = batch_seqs.to(device)   # (B, max_seq_len)
                batch_pos  = batch_pos.to(device)    # (B,)

                # 动态负采样
                neg_list = []
                for uid in batch_uids:
                    pos_set    = user_pos_set.get(int(uid), set())
                    candidates = all_movie_arr[~np.isin(all_movie_arr, list(pos_set))]
                    if len(candidates) == 0:
                        candidates = all_movie_arr
                    sampled = rng.choice(
                        candidates,
                        size=neg_sample_ratio,
                        replace=len(candidates) < neg_sample_ratio
                    )
                    neg_list.append(sampled)

                neg_arr    = torch.LongTensor(np.array(neg_list)).to(device)  # (B, neg_ratio)
                B          = batch_seqs.size(0)
                neg_flat   = neg_arr.view(-1)                                  # (B*neg_ratio,)
                seq_repeat = batch_seqs.repeat_interleave(neg_sample_ratio, dim=0)

                pos_scores = self(batch_seqs, batch_pos)
                pos_labels = torch.ones(B, device=device)
                neg_scores = self(seq_repeat, neg_flat)
                neg_labels = torch.zeros(B * neg_sample_ratio, device=device)

                loss = criterion(torch.cat([pos_scores, neg_scores]),
                                 torch.cat([pos_labels, neg_labels]))
                loss = loss + self.get_l2_loss()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item() * (B + B * neg_sample_ratio)
                total_cnt  += (B + B * neg_sample_ratio)

                # 实时在 batch 进度条后缀显示当前滑动 loss
                batch_bar.set_postfix(loss=f"{total_loss / total_cnt:.4f}")

            batch_bar.close()
            train_avg_loss = total_loss / total_cnt

            # ── 验证 BCE ──────────────────────────────
            val_str    = ""
            early_stop = False
            if val_samples is not None:
                val_avg_loss = self._compute_val_bce(val_samples, batch_size, criterion, device)
                val_str = f"  val={val_avg_loss:.4f}"

                if early_stopping_patience > 0:
                    if val_avg_loss < best_val_loss - early_stopping_min_delta:
                        best_val_loss    = val_avg_loss
                        patience_counter = 0
                        best_state       = {k: v.clone() for k, v in self.state_dict().items()}
                        val_str         += " ✓"
                    else:
                        patience_counter += 1
                        val_str         += f" (p:{patience_counter}/{early_stopping_patience})"
                        if patience_counter >= early_stopping_patience:
                            early_stop = True

            # 更新 epoch 进度条后缀
            epoch_bar.set_postfix_str(
                f"train={train_avg_loss:.4f}{val_str}"
            )

            if early_stop:
                tqdm.write(f"  Early Stopping! 最佳 Val BCE: {best_val_loss:.4f}")
                break

        # 恢复最佳模型
        if best_state is not None:
            self.load_state_dict(best_state)
            print(f"  训练完成, 使用最佳模型 (Val BCE: {best_val_loss:.4f})")
        print("SASRec 模型训练完成")

    def _compute_val_bce(self, val_samples, batch_size, criterion, device):
        """
        在验证样本上计算平均 BCE Loss (无梯度)

        val_samples: _SASRecTrainDataset — (uid, seq, pos_mid) 全为正例
        负例在此处固定采样一次 (seed=0), 保证每 epoch 验证集一致
        """
        self.eval()
        rng_val = np.random.default_rng(0)

        val_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=False)
        total_loss = 0.0
        total_cnt  = 0

        with torch.no_grad():
            for _, batch_seqs, batch_pos in val_loader:
                batch_seqs = batch_seqs.to(device)
                batch_pos  = batch_pos.to(device)
                B          = batch_seqs.size(0)

                pos_scores = self(batch_seqs, batch_pos)
                pos_labels = torch.ones(B, device=device)
                loss       = criterion(pos_scores, pos_labels)

                total_loss += loss.item() * B
                total_cnt  += B

        self.train()
        return total_loss / total_cnt if total_cnt > 0 else 0.0

    def predict(self, item_seqs, target_items):
        """
        批量预测 (numpy 接口)

        Args:
            item_seqs:    np.ndarray (N, max_seq_len) — 用户历史序列
            target_items: np.ndarray (N,) — 候选物品 ID

        Returns:
            scores: np.ndarray (N,)
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            seq_t    = torch.LongTensor(item_seqs).to(device)
            item_t   = torch.LongTensor(target_items).to(device)
            scores   = self(seq_t, item_t)
        return scores.cpu().numpy()

    def save(self, filepath='models/sasrec_model.pth'):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_items':    self.num_items,
            'hidden_dim':   self.hidden_dim,
            'max_seq_len':  self.max_seq_len,
            'num_blocks':   self.num_blocks,
            'num_heads':    self.num_heads,
            'dropout_rate': self.dropout_rate,
            'item_emb_l2':  self.item_emb_l2,
        }
        torch.save(checkpoint, filepath)
        print(f"  SASRec 模型已保存至: {filepath}")

    @staticmethod
    def load(filepath='models/sasrec_model.pth',
             num_items=3953, hidden_dim=64, max_seq_len=50,
             num_blocks=2, num_heads=2, dropout_rate=0.2, item_emb_l2=0.0):
        """加载模型"""
        if not os.path.exists(filepath):
            return None
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        model = SASRecModel(
            num_items=checkpoint.get('num_items',    num_items),
            hidden_dim=checkpoint.get('hidden_dim',  hidden_dim),
            max_seq_len=checkpoint.get('max_seq_len', max_seq_len),
            num_blocks=checkpoint.get('num_blocks',  num_blocks),
            num_heads=checkpoint.get('num_heads',    num_heads),
            dropout_rate=checkpoint.get('dropout_rate', dropout_rate),
            item_emb_l2=checkpoint.get('item_emb_l2', item_emb_l2),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  已加载 SASRec 模型: {filepath}")
        return model


# ──────────────────────────────────────────
# 4. 辅助工具
# ──────────────────────────────────────────

def _pad_sequence(seq, max_len):
    """将序列 padding/截断到 max_len (右填 0, 保留最新的 max_len 个交互)"""
    seq = list(seq)
    if len(seq) >= max_len:
        return seq[-max_len:]
    else:
        return [0] * (max_len - len(seq)) + seq


def _build_train_samples(user_train_seqs, max_seq_len):
    """
    从训练序列构建正例样本
    每条序列的每个位置 t (t≥1) 产生一个训练样本:
        输入: seq[:t]  (padding 到 max_seq_len)
        正例: seq[t]

    Args:
        user_train_seqs: dict {uid -> list[mid]}
        max_seq_len: 序列最大长度

    Returns:
        list of (uid, seq_array, pos_mid)
    """
    samples = []
    for uid, seq in user_train_seqs.items():
        if len(seq) < 2:
            continue
        for t in range(1, len(seq)):
            hist   = seq[:t]
            pos    = seq[t]
            padded = _pad_sequence(hist, max_seq_len)
            samples.append((uid, np.array(padded, dtype=np.int32), pos))
    return samples


def _build_val_samples(user_val_pos, user_train_seqs, user_pos_set,
                       all_movie_arr, max_seq_len):
    """
    构建验证样本 Dataset (只含正例, BCE 验证时仅用正例计算 loss)

    Args:
        user_val_pos:    dict {uid -> pos_movie_id}
        user_train_seqs: dict {uid -> list[movie_id]}
        user_pos_set:    dict {uid -> set(movie_id)}  — 用于过滤 (保留接口一致性)
        all_movie_arr:   np.ndarray — 全量电影 ID
        max_seq_len:     序列截断长度

    Returns:
        _SASRecTrainDataset
    """
    samples = []
    for uid, pos_mid in user_val_pos.items():
        seq    = user_train_seqs.get(uid, [])
        padded = _pad_sequence(seq, max_seq_len)
        samples.append((uid, np.array(padded, dtype=np.int32), pos_mid))
    return _SASRecTrainDataset(samples)


class _SASRecTrainDataset(Dataset):
    def __init__(self, samples):
        """
        samples: list of (uid, seq_array[max_seq_len], pos_mid)
        """
        self.uids = torch.LongTensor([s[0] for s in samples])
        self.seqs = torch.LongTensor(np.stack([s[1] for s in samples]))   # (N, L)
        self.pos  = torch.LongTensor([s[2] for s in samples])

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        return self.uids[idx], self.seqs[idx], self.pos[idx]


# ──────────────────────────────────────────
# 5. 评估适配器 (与 evaluation.py 接口兼容)
# ──────────────────────────────────────────

class SASRecPredictor:
    """
    包装 SASRecModel, 提供与 WideAndDeep 相同的 predict(X) 接口,
    供 evaluation.py 中的 calculate_hr_and_ndcg_at_k 调用.

    评估时特征格式:
        X: (N, 2) — [user_id, movie_id]  (SASRec 只需要 user 历史和候选 item)

    预测流程:
        1. 从 user_id 查出历史序列
        2. 对每条 (user_id, movie_id) 构造 (seq, target) 对
        3. 用 SASRecModel.predict() 返回分数
    """

    def __init__(self, model: SASRecModel, user_seq_map: dict, max_seq_len: int):
        """
        Args:
            model:        训练好的 SASRecModel
            user_seq_map: dict {user_id -> list[movie_id]} 用户历史序列 (训练集)
            max_seq_len:  序列截断长度
        """
        self.model       = model
        self.user_seq    = user_seq_map
        self.max_seq_len = max_seq_len

    def predict(self, X):
        """
        Args:
            X: np.ndarray (N, ≥2) — 第0列 user_id, 第1列 movie_id

        Returns:
            scores: np.ndarray (N,)
        """
        user_ids   = X[:, 0].astype(int)
        movie_ids  = X[:, 1].astype(int)

        seqs = np.array([
            _pad_sequence(self.user_seq.get(int(uid), []), self.max_seq_len)
            for uid in user_ids
        ], dtype=np.int32)

        return self.model.predict(seqs, movie_ids)


# ──────────────────────────────────────────
# 6. 训练入口函数 (与 model_wide_deep.py 风格一致)
# ──────────────────────────────────────────

def train_sasrec_model(user_train_seqs, all_movie_ids,
                       user_val_pos=None,
                       num_items=3953,
                       hidden_dim=64, max_seq_len=50,
                       num_blocks=2, num_heads=2, dropout_rate=0.2, item_emb_l2=0.0,
                       epochs=30, batch_size=256, learning_rate=1e-3,
                       neg_sample_ratio=1, verbose=True,
                       early_stopping_patience=5, early_stopping_min_delta=1e-4):
    """
    训练 SASRec 模型

    Args:
        user_train_seqs: dict {user_id -> list[movie_id]} — 训练序列 (按时间排序)
        all_movie_ids:   list — 全量电影 ID
        user_val_pos:    dict {user_id -> movie_id} — 验证集正例 (可选, 用于 Val BCE Early Stopping)
        num_items:       电影总数
        hidden_dim:      Embedding/隐层维度 (默认 64)
        max_seq_len:     序列最大长度 (默认 50)
        num_blocks:      Transformer Block 数量 (默认 2)
        num_heads:       多头注意力头数 (默认 2)
        dropout_rate:    Dropout 比率 (默认 0.2)
        item_emb_l2:     Item Embedding L2 正则化系数
        epochs:          训练轮数
        batch_size:      批大小
        learning_rate:   学习率
        neg_sample_ratio: 正负例比 (默认 1:1)
        verbose:         打印详情
        early_stopping_patience: Early Stopping 耐心值
        early_stopping_min_delta: 最小改善阈值

    Returns:
        SASRecModel
    """
    print("\n[4/5] 训练 SASRec 模型...")
    print(f"  hidden_dim={hidden_dim}, max_seq_len={max_seq_len}")
    print(f"  num_blocks={num_blocks}, num_heads={num_heads}, dropout={dropout_rate}")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    model = SASRecModel(
        num_items=num_items,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        item_emb_l2=item_emb_l2,
    )

    model.fit(
        user_train_seqs=user_train_seqs,
        all_movie_ids=all_movie_ids,
        user_val_pos=user_val_pos,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        neg_sample_ratio=neg_sample_ratio,
        verbose=verbose,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    return model
