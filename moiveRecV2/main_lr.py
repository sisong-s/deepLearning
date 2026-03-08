"""
MovieLens 1M Baseline测试 - 逻辑回归
数据集划分: 按时间排序 (前80% train, 80%-90% val, 后10% test)
评估指标: RMSE, NDCG@10

主程序入口,整合所有模块
"""

import warnings
warnings.filterwarnings('ignore')

from data_pipeline import load_data, split_data_by_time
from feature_engineering import prepare_features, convert_to_classification, proba_to_rating
from model_training import train_logistic_regression, predict, save_model, load_model
from evaluation import evaluate_model


def main(apply_augmentation=False, augment_config=None):
    """
    主函数
    
    Args:
        apply_augmentation: 是否应用数据增强
        augment_config: 数据增强配置
    """
    print("=" * 80)
    print("MovieLens 1M Baseline测试 - 逻辑回归 (LR)")
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
    
    # 将评分转换为分类标签 (0-4)
    y_train_class, y_val_class, y_test_class = convert_to_classification(
        y_train, y_val, y_test
    )
    
    # 4. 训练或加载模型
    # 根据是否使用数据增强,使用不同的模型文件
    if apply_augmentation:
        config_str = f"{augment_config.get('target_distribution', 'uniform')}_{augment_config.get('augment_ratio', 0.5)}"
        model_path = f'models/lr_model_aug_{config_str}.pkl'
    else:
        model_path = 'models/lr_model.pkl'
    
    lr_model = load_model(model_path)
    
    if lr_model is None:
        # 模型不存在,进行训练
        lr_model = train_logistic_regression(X_train, y_train_class)
        # 保存训练好的模型
        save_model(lr_model, model_path)
    else:
        print(f"[4/5] 使用已保存的模型: {model_path}")
    
    # 5. 预测
    y_val_proba = predict(lr_model, X_val)
    y_test_proba = predict(lr_model, X_test)
    
    # 将概率转换为期望评分
    y_val_pred = proba_to_rating(y_val_proba)
    y_test_pred = proba_to_rating(y_test_proba)
    
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
    import sys
    
    # 支持命令行参数控制数据增强
    if len(sys.argv) > 1 and sys.argv[1] == '--augment':
        # 示例用法:
        # python main_lr.py --augment
        # python main_lr.py --augment uniform 0.5
        # python main_lr.py --augment balanced 1.0
        
        target_dist = sys.argv[2] if len(sys.argv) > 2 else 'uniform'
        augment_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        
        augment_config = {
            'target_distribution': target_dist,
            'augment_ratio': augment_ratio
        }
        
        print(f">>> 启用数据增强: {target_dist} distribution, ratio={augment_ratio}")
        main(apply_augmentation=True, augment_config=augment_config)
    else:
        # 默认不使用数据增强
        main(apply_augmentation=False)
