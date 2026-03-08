"""
MovieLens 1M Baseline测试 - Wide&Deep模型 (PyTorch联合训练版本)
数据集划分: 按时间排序 (前80% train, 80%-90% val, 后10% test)
评估指标: RMSE, NDCG@10

Wide部分: 线性层
Deep部分: DNN
联合训练: Wide和Deep通过反向传播共同优化
"""

import warnings
warnings.filterwarnings('ignore')

from data_pipeline import load_data, split_data_by_time
from feature_engineering import prepare_features
from model_wide_deep import train_wide_deep_model, predict, WideAndDeepModel
from evaluation import evaluate_model


def main(apply_augmentation=False, augment_config=None):
    """
    主函数
    
    Args:
        apply_augmentation: 是否应用数据增强
        augment_config: 数据增强配置
    """
    print("=" * 80)
    print("MovieLens 1M Baseline测试 - Wide&Deep模型 (PyTorch联合训练)")
    if apply_augmentation:
        print(f"数据增强: 启用 {augment_config}")
    print("=" * 80)
    
    # 1. 加载数据
    ratings, users, movies = load_data()
    
    # 2. 按时间划分数据集
    train_data, val_data, test_data = split_data_by_time(ratings)
    
    # 3. 准备特征(包含数据增强)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_stats = prepare_features(
        train_data, val_data, test_data, users, movies,
        apply_augmentation=apply_augmentation,
        augment_config=augment_config
    )
    
    # 从feature_stats中提取各类别的数量(已包含OOV处理)
    num_users = feature_stats['UserID']['max_value']  # 训练集最大UserID + 1 (UNK)
    num_movies = feature_stats['MovieID']['max_value']  # 训练集最大MovieID + 1 (UNK)
    num_ages = feature_stats['Age']['num_unique']
    num_occupations = feature_stats['Occupation']['num_unique']
    num_movie_years = feature_stats['MoiveYear']['num_unique']
    num_rate_years = feature_stats['RateYear']['num_unique']
    num_stat_features = 6  # 统计特征数量: 用户3个 + 电影3个
    
    print(f"模型参数配置:")
    print(f"  num_users: {num_users} (包含UNK)")
    print(f"  num_movies: {num_movies} (包含UNK)")
    print(f"  num_ages: {num_ages}")
    print(f"  num_occupations: {num_occupations}")
    print(f"  num_movie_years: {num_movie_years}")
    print(f"  num_rate_years: {num_rate_years}")
    print(f"  num_stat_features: {num_stat_features}")
    
    # 4. 训练或加载模型
    # 根据是否使用数据增强,使用不同的模型文件
    if apply_augmentation:
        config_str = f"{augment_config.get('target_distribution', 'uniform')}_{augment_config.get('augment_ratio', 0.5)}"
        model_path = f'models/wide_deep_model_aug_{config_str}.pth'
    else:
        model_path = 'models/wide_deep_model.pth'
    
    hidden_units = [64, 32]
    dropout_rate = 0.3  # Dropout比率
    wide_l2_reg = 0.01  # Wide侧L2正则化系数
    early_stopping_patience = 4  # Early Stopping耐心值
    early_stopping_min_delta = 0.0001  # Early Stopping最小改善阈值
    
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
        # 模型不存在,进行训练
        wd_model = train_wide_deep_model(
            X_train, y_train, 
            X_val, y_val,
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
            early_stopping_min_delta=early_stopping_min_delta
        )
        # 保存训练好的模型
        wd_model.save(model_path)
    else:
        print(f"[4/5] 使用已保存的模型: {model_path}")
    
    # 5. 预测 (直接输出评分值,无需转换)
    y_val_pred = predict(wd_model, X_val)
    y_test_pred = predict(wd_model, X_test)
    
    # 6. 评估模型
    metrics = evaluate_model(
        y_val, y_val_pred, val_data['UserID'].values,
        y_test, y_test_pred, test_data['UserID'].values
    )
    
    print("\n" + "=" * 80)
    print("实验完成!")
    print("=" * 80)
    
    return metrics


if __name__ == "__main__":
    # 默认启用50%均匀分布数据增强
    augment_config = {
        'target_distribution': 'uniform',
        'augment_ratio': 0.5
    }
    main(apply_augmentation=True, augment_config=augment_config)
