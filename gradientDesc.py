# 假数据（只有4个点）
x_data = [1, 2, 3, 4]
y_data = [3.1, 5.2, 6.8, 9.0]   # 有点噪声

# 前向计算：预测值
def predict(x, w, b):
    return w * x + b

# 均方误差（损失函数）
# def compute_loss(w, b):
#     total = 0
#     for x, y in zip(x_data, y_data):
#         pred = predict(x, w, b)
#         total += (pred - y) ** 2
#     return total / len(x_data)

# 手动求导（对 w 和 b 的偏导）
def compute_gradient(w, b):
    dw = 0
    db = 0
    for x, y in zip(x_data, y_data):
        pred = predict(x, w, b)
        error = pred - y
        dw += 2 * error * x     # ∂L/∂w
        db += 2 * error * 1     # ∂L/∂b
    n = len(x_data)
    return dw / n, db / n

# -----------------------
#     梯度下降
# -----------------------

w = 0.0    # 初始权重
b = 0.0    # 初始偏置
lr = 0.01
epochs = 100

# print("初始状态 →  w: %.4f,  b: %.4f,  loss: %.6f\n" % (w, b, compute_loss(w, b)))

for epoch in range(epochs):
    dw, db = compute_gradient(w, b)
    
    w -= lr * dw
    b -= lr * db
    
    # if (epoch + 1) % 20 == 0 or epoch == epochs-1:
    #     loss = compute_loss(w, b)
    #     print(f"第 {epoch+1:3d} 轮 →  w: {w:6.4f},  b: {b:6.4f},  loss: {loss:.6f}")

print("\n最终结果：")
print(f"w ≈ {w:.4f}   (真实 ≈ 2.0)")
print(f"b ≈ {b:.4f}   (真实 ≈ 1.0)")