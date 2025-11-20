#!/usr/bin/env python3
"""
独立的显存分析脚本
用于分析模型各部分的显存占用情况
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from memory_monitor import MemoryMonitor
import argparse

def analyze_model_memory(model_name: str, use_lora: bool = True, lora_rank: int = 8):
    """分析模型显存占用"""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化显存监控器
    memory_monitor = MemoryMonitor(device)
    
    print("\n" + "="*60)
    print("开始分析模型显存占用")
    print("="*60)
    
    # 1. 初始状态
    memory_monitor.take_snapshot("initial")
    memory_monitor.print_memory_summary()
    
    # 2. 加载tokenizer
    print("\n正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    memory_monitor.take_snapshot("tokenizer_loaded")
    print("Tokenizer加载完成")
    memory_monitor.compare_snapshots("initial", "tokenizer_loaded")
    
    # 3. 加载基础模型
    print("\n正在加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    memory_monitor.take_snapshot("base_model_loaded")
    print("基础模型加载完成")
    memory_monitor.compare_snapshots("tokenizer_loaded", "base_model_loaded")
    memory_monitor.print_memory_summary(model)
    
    # 4. 添加LoRA (如果需要)
    if use_lora:
        print(f"\n正在添加LoRA (rank={lora_rank})...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        memory_monitor.take_snapshot("lora_added")
        print("LoRA添加完成")
        memory_monitor.compare_snapshots("base_model_loaded", "lora_added")
        memory_monitor.print_memory_summary(model)
    
    # 5. 移动到GPU
    print(f"\n正在将模型移动到 {device}...")
    model = model.to(device)
    memory_monitor.take_snapshot("model_on_device")
    print("模型移动完成")
    memory_monitor.compare_snapshots("lora_added" if use_lora else "base_model_loaded", "model_on_device")
    memory_monitor.print_memory_summary(model)
    
    # 6. 模拟训练状态
    print("\n模拟训练状态 (创建优化器)...")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
    memory_monitor.take_snapshot("optimizer_created")
    print("优化器创建完成")
    memory_monitor.compare_snapshots("model_on_device", "optimizer_created")
    
    # 7. 模拟前向传播
    print("\n模拟前向传播...")
    model.train()
    
    # 创建模拟输入
    batch_size = 1
    seq_length = 256
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    labels = input_ids.clone()
    
    memory_monitor.take_snapshot("before_forward")
    
    # 前向传播
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    memory_monitor.take_snapshot("after_forward")
    print("前向传播完成")
    memory_monitor.compare_snapshots("before_forward", "after_forward")
    
    # 8. 模拟反向传播
    print("\n模拟反向传播...")
    loss.backward()
    memory_monitor.take_snapshot("after_backward")
    print("反向传播完成")
    memory_monitor.compare_snapshots("after_forward", "after_backward")
    
    # 最终显存状态
    print("\n" + "="*60)
    print("最终显存状态分析")
    print("="*60)
    memory_monitor.print_memory_summary(model)
    
    # 显示所有快照对比
    print("\n" + "="*60)
    print("各阶段显存变化总结")
    print("="*60)
    snapshots = [
        ("initial", "tokenizer_loaded", "加载Tokenizer"),
        ("tokenizer_loaded", "base_model_loaded", "加载基础模型"),
    ]
    
    if use_lora:
        snapshots.append(("base_model_loaded", "lora_added", "添加LoRA"))
        snapshots.append(("lora_added", "model_on_device", "移动到GPU"))
    else:
        snapshots.append(("base_model_loaded", "model_on_device", "移动到GPU"))
    
    snapshots.extend([
        ("model_on_device", "optimizer_created", "创建优化器"),
        ("before_forward", "after_forward", "前向传播"),
        ("after_forward", "after_backward", "反向传播")
    ])
    
    for snap1, snap2, desc in snapshots:
        print(f"\n{desc}:")
        memory_monitor.compare_snapshots(snap1, snap2)

def main():
    parser = argparse.ArgumentParser(description="分析模型显存占用")
    parser.add_argument("--model", default="./modelscope/Qwen/Qwen2.5-1.5B-Instruct", 
                       help="模型路径")
    parser.add_argument("--no-lora", action="store_true", help="不使用LoRA")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    
    args = parser.parse_args()
    
    analyze_model_memory(
        model_name=args.model,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank
    )

if __name__ == "__main__":
    main()