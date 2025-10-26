import torch

# 测试CUDA是否可用
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 数量:", torch.cuda.device_count())
    print("当前 GPU:", torch.cuda.get_device_name(0))
    
    # 选择设备
    device = torch.device("cuda:0")
    print("使用设备:", device)
else:
    device = torch.device("cpu")
    print("使用设备:", device)

print(torch.__version__)
# 验证 CUDA 是否可用
print("CUDA 可用状态:", torch.cuda.is_available())  # 应输出 True
# 查看 PyTorch 绑定的 CUDA 版本（显示 11.8，正常）
print("PyTorch 绑定 CUDA 版本:", torch.version.cuda)
# 查看系统显卡驱动支持的最高 CUDA 版本（显示 12.6，正常）
print("显卡驱动支持最高 CUDA 版本:", torch.cuda.get_device_capability())  # 输出显卡算力，间接反映驱动兼容性