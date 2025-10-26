import torch

def simple_cuda_test():
    """简单的CUDA测试"""
    print("=" * 40)
    print("简单 CUDA 测试")
    print("=" * 40)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print("✅ CUDA 可用")
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        print(f"   当前 GPU: {torch.cuda.get_device_name(0)}")
        
        # 创建张量并移动到GPU
        device = torch.device("cuda:0")
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        
        # 执行简单运算
        z = x + y
        
        print(f"   张量设备: {x.device}")
        print(f"   运算结果:\n{z}")
        print("✅ GPU 运算成功")
        
    else:
        print("❌ CUDA 不可用，使用 CPU")
        device = torch.device("cpu")
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        print(f"   CPU 运算结果:\n{z}")

if __name__ == "__main__":
    simple_cuda_test()