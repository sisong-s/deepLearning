import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义一个简单的 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)   # 第一层
        self.relu = nn.ReLU()                              # 激活函数
        self.layer2 = nn.Linear(hidden_size, num_classes)  # 输出层
    
    def forward(self, x):
        x = x.view(x.size(0), -1)    # 把图片展平 (batch_size, 28*28)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 2. 准备数据（用 MNIST 举例）
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 把图片（原来是 PIL.Image，值 0~255） → 变成 torch.Tensor（值变成 0.0~1.0），并且自动把通道顺序从 (H×W×C) 改成 (C×H×W)，这是 PyTorch 的标准格式
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# shuffle 随机
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. 创建模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP(input_size=28*28, hidden_size=128, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()                  # 分类任务常用交叉熵
optimizer = optim.SGD(model.parameters(), lr=0.01) # 简单用 SGD 也可以换 Adam

# 4. 训练循环（只跑 5 个 epoch 演示）
model.train()
for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}")

# 5. 测试准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试准确率: {100 * correct / total:.2f}%")