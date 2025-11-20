import torch
import torch.nn as nn

# 超简单两层 MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)   # 输入100维 -> 50维 Fully Connected layer
        self.fc2 = nn.Linear(50, 10)    # 50维 -> 10类
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 测试一下
model = MLP()
x = torch.randn(32, 100)   # 假数据：32个样本，每个100维
output = model(x)
print(output.shape)  # torch.Size([32, 10])

# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)   # SGD 也能收敛得很清楚

# # 3. 训练 500 次（数据极少，几百次就完全收敛）
# for epoch in range(501):
#     optimizer.zero_grad()
#     pred = model(X)
#     loss = criterion(pred, Y)
#     loss.backward()
#     optimizer.step()