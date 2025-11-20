import torch
import gc
from typing import Dict, List, Optional
import psutil
import os
from collections import defaultdict

class MemoryMonitor:
    """GPU和CPU内存监控工具"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == 'cuda'
        self.memory_snapshots = []
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """获取GPU显存信息 (MB)"""
        if not self.is_cuda:
            return {"allocated": 0, "cached": 0, "reserved": 0, "free": 0, "total": 0}
            
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        cached = torch.cuda.memory_cached(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        
        # 获取GPU总显存
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
        free_memory = total_memory - reserved
        
        return {
            "allocated": allocated,
            "cached": cached, 
            "reserved": reserved,
            "free": free_memory,
            "total": total_memory
        }
    
    def get_cpu_memory_info(self) -> Dict[str, float]:
        """获取CPU内存信息 (MB)"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss / 1024**2,  # 物理内存
            "vms": memory_info.vms / 1024**2,  # 虚拟内存
            "percent": process.memory_percent()
        }
    
    def analyze_model_memory(self, model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
        """分析模型各部分的显存占用"""
        memory_breakdown = defaultdict(lambda: {"parameters": 0, "buffers": 0, "total": 0})
        
        # 分析参数显存占用
        for name, param in model.named_parameters():
            if param.device == self.device:
                param_memory = param.numel() * param.element_size() / 1024**2
                
                # 根据参数名称分类
                if "lora" in name.lower():
                    category = "LoRA"
                elif any(x in name for x in ["embed", "wte", "word_embeddings"]):
                    category = "Embeddings"
                elif any(x in name for x in ["attention", "attn", "q_proj", "k_proj", "v_proj", "o_proj"]):
                    category = "Attention"
                elif any(x in name for x in ["mlp", "ffn", "gate_proj", "up_proj", "down_proj"]):
                    category = "MLP/FFN"
                elif "norm" in name or "ln" in name:
                    category = "LayerNorm"
                else:
                    category = "Other"
                
                memory_breakdown[category]["parameters"] += param_memory
        
        # 分析缓冲区显存占用
        for name, buffer in model.named_buffers():
            if buffer.device == self.device:
                buffer_memory = buffer.numel() * buffer.element_size() / 1024**2
                
                if any(x in name for x in ["attention", "attn"]):
                    category = "Attention"
                elif "norm" in name or "ln" in name:
                    category = "LayerNorm"
                else:
                    category = "Other"
                
                memory_breakdown[category]["buffers"] += buffer_memory
        
        # 计算总计
        for category in memory_breakdown:
            memory_breakdown[category]["total"] = (
                memory_breakdown[category]["parameters"] + 
                memory_breakdown[category]["buffers"]
            )
        
        return dict(memory_breakdown)
    
    def get_tensor_memory_by_type(self) -> Dict[str, float]:
        """按张量类型分析显存占用"""
        if not self.is_cuda:
            return {}
            
        tensor_memory = defaultdict(float)
        
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and obj.device == self.device:
                tensor_size = obj.numel() * obj.element_size() / 1024**2
                dtype_name = str(obj.dtype).replace('torch.', '')
                tensor_memory[dtype_name] += tensor_size
        
        return dict(tensor_memory)
    
    def print_memory_summary(self, model: Optional[torch.nn.Module] = None, 
                           step: Optional[int] = None, epoch: Optional[int] = None):
        """打印详细的显存使用情况"""
        print("\n" + "="*60)
        if step is not None and epoch is not None:
            print(f"Memory Summary - Epoch {epoch}, Step {step}")
        else:
            print("Memory Summary")
        print("="*60)
        
        # GPU显存信息
        gpu_info = self.get_gpu_memory_info()
        if self.is_cuda:
            print(f"GPU Memory (MB):")
            print(f"  Allocated: {gpu_info['allocated']:.1f}")
            print(f"  Cached:    {gpu_info['cached']:.1f}")
            print(f"  Reserved:  {gpu_info['reserved']:.1f}")
            print(f"  Free:      {gpu_info['free']:.1f}")
            print(f"  Total:     {gpu_info['total']:.1f}")
            print(f"  Usage:     {gpu_info['reserved']/gpu_info['total']*100:.1f}%")
        
        # CPU内存信息
        cpu_info = self.get_cpu_memory_info()
        print(f"\nCPU Memory (MB):")
        print(f"  RSS:       {cpu_info['rss']:.1f}")
        print(f"  VMS:       {cpu_info['vms']:.1f}")
        print(f"  Percent:   {cpu_info['percent']:.1f}%")
        
        # 模型各部分显存占用
        if model is not None:
            model_memory = self.analyze_model_memory(model)
            print(f"\nModel Memory Breakdown (MB):")
            total_model_memory = 0
            for category, memory_info in sorted(model_memory.items()):
                if memory_info["total"] > 0:
                    print(f"  {category:12s}: {memory_info['total']:8.1f} "
                          f"(Params: {memory_info['parameters']:6.1f}, "
                          f"Buffers: {memory_info['buffers']:6.1f})")
                    total_model_memory += memory_info["total"]
            print(f"  {'Total':12s}: {total_model_memory:8.1f}")
        
        # 按数据类型分析
        tensor_memory = self.get_tensor_memory_by_type()
        if tensor_memory:
            print(f"\nTensor Memory by Type (MB):")
            for dtype, memory in sorted(tensor_memory.items()):
                if memory > 0.1:  # 只显示大于0.1MB的
                    print(f"  {dtype:12s}: {memory:8.1f}")
        
        print("="*60)
    
    def log_memory_to_tensorboard(self, writer, step: int, model: Optional[torch.nn.Module] = None):
        """将显存信息记录到TensorBoard"""
        gpu_info = self.get_gpu_memory_info()
        cpu_info = self.get_cpu_memory_info()
        
        # 记录GPU显存
        if self.is_cuda:
            writer.add_scalar('Memory/GPU_Allocated_MB', gpu_info['allocated'], step)
            writer.add_scalar('Memory/GPU_Cached_MB', gpu_info['cached'], step)
            writer.add_scalar('Memory/GPU_Reserved_MB', gpu_info['reserved'], step)
            writer.add_scalar('Memory/GPU_Usage_Percent', 
                            gpu_info['reserved']/gpu_info['total']*100, step)
        
        # 记录CPU内存
        writer.add_scalar('Memory/CPU_RSS_MB', cpu_info['rss'], step)
        writer.add_scalar('Memory/CPU_Percent', cpu_info['percent'], step)
        
        # 记录模型各部分显存
        if model is not None:
            model_memory = self.analyze_model_memory(model)
            for category, memory_info in model_memory.items():
                if memory_info["total"] > 0:
                    writer.add_scalar(f'Memory/Model_{category}_MB', 
                                    memory_info["total"], step)
    
    def clear_cache(self):
        """清理显存缓存"""
        if self.is_cuda:
            torch.cuda.empty_cache()
        gc.collect()
    
    def take_snapshot(self, name: str):
        """保存当前显存状态快照"""
        snapshot = {
            "name": name,
            "gpu": self.get_gpu_memory_info(),
            "cpu": self.get_cpu_memory_info()
        }
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def compare_snapshots(self, name1: str, name2: str):
        """比较两个快照的显存差异"""
        snap1 = next((s for s in self.memory_snapshots if s["name"] == name1), None)
        snap2 = next((s for s in self.memory_snapshots if s["name"] == name2), None)
        
        if not snap1 or not snap2:
            print("Snapshot not found!")
            return
        
        print(f"\nMemory Difference: {name2} - {name1}")
        print("-" * 40)
        
        if self.is_cuda:
            gpu_diff = snap2["gpu"]["allocated"] - snap1["gpu"]["allocated"]
            print(f"GPU Allocated: {gpu_diff:+.1f} MB")
        
        cpu_diff = snap2["cpu"]["rss"] - snap1["cpu"]["rss"]
        print(f"CPU RSS: {cpu_diff:+.1f} MB")