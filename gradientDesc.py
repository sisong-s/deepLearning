import numpy as np

# 1. 定义梯度下降函数（一元线性回归为例）
def gradient_descent(x, y, lr, epochs):
    # 初始化参数 w、b（随便给初始值，比如0）
    w = 0.0
    b = 0.0
    n = len(x)  # 样本数量
    
    for i in range(epochs):
        # 前向计算：预测值 y_pred = w*x + b
        y_pred = w * x + b
        
        # 计算损失（可选，用于观察收敛情况）
        loss = np.mean((y - y_pred) **2)
        
        # 计算梯度（核心！）
        dw = (-2/n) * np.sum(x * (y - y_pred))  # loss对w的梯度
        db = (-2/n) * np.sum(y - y_pred)        # loss对b的梯度
        
        # 更新参数（核心！负梯度方向）
        w -= lr * dw
        b -= lr * db
        
        # 每100轮打印一次，看损失是否下降（面试时可省略，口述即可）
        if i % 100 == 0:
            print(f"Epoch {i}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")
    
    return w, b

# 2. 测试代码（生成模拟数据）
if __name__ == "__main__":
    # 生成真实数据：y = 2x + 1 + 少量噪声
    x = np.array([1, 2, 3, 4, 5])
    y = 2 * x + 1 + np.random.randn(5) * 0.01  # 加噪声更贴近真实场景
    
    # 运行梯度下降
    w_final, b_final = gradient_descent(x, y, lr=0.01, epochs=2000)
    print(f"\n最终参数：w={w_final:.4f}, b={b_final:.4f}")  # 接近2和1