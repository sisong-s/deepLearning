#!/usr/bin/env python3
"""
原生FSDP训练启动脚本
支持单机多卡和多机多卡训练
"""

import os
import subprocess
import sys
import torch


def run_single_node_multi_gpu(script_path, num_gpus=None):
    """单机多卡训练"""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus < 2:
        print("警告: 检测到GPU数量少于2个，FSDP主要用于多GPU训练")
        print(f"当前可用GPU数量: {num_gpus}")
        
    print(f"启动单机{num_gpus}卡FSDP训练...")
    
    # 使用torchrun启动分布式训练
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(num_gpus),
        "--nnodes", "1",
        "--node_rank", "0",
        "--master_addr", "localhost",
        "--master_port", "12355",
        script_path
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"训练失败，错误码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("训练被用户中断")
        return False
    
    return True


def run_multi_node(script_path, nnodes, node_rank, master_addr, master_port, nproc_per_node):
    """多机多卡训练"""
    print(f"启动多机多卡FSDP训练...")
    print(f"节点数: {nnodes}, 当前节点: {node_rank}, 主节点地址: {master_addr}:{master_port}")
    
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(nproc_per_node),
        "--nnodes", str(nnodes),
        "--node_rank", str(node_rank),
        "--master_addr", master_addr,
        "--master_port", str(master_port),
        script_path
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"训练失败，错误码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("训练被用户中断")
        return False
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FSDP训练启动器")
    parser.add_argument("--script", default="fsdp_native_training.py", 
                       help="训练脚本路径 (默认: fsdp_native_training.py)")
    parser.add_argument("--num_gpus", type=int, default=None,
                       help="使用的GPU数量 (默认: 自动检测)")
    
    # 多机训练参数
    parser.add_argument("--nnodes", type=int, default=1,
                       help="节点数量 (默认: 1)")
    parser.add_argument("--node_rank", type=int, default=0,
                       help="当前节点编号 (默认: 0)")
    parser.add_argument("--master_addr", default="localhost",
                       help="主节点地址 (默认: localhost)")
    parser.add_argument("--master_port", type=int, default=12355,
                       help="主节点端口 (默认: 12355)")
    parser.add_argument("--nproc_per_node", type=int, default=None,
                       help="每个节点的进程数 (默认: 自动检测GPU数量)")
    
    args = parser.parse_args()
    
    # 检查脚本是否存在
    if not os.path.exists(args.script):
        print(f"错误: 训练脚本 {args.script} 不存在")
        return False
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，无法进行GPU训练")
        return False
    
    print(f"检测到 {torch.cuda.device_count()} 个GPU")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 根据参数选择训练模式
    if args.nnodes == 1:
        # 单机多卡
        success = run_single_node_multi_gpu(args.script, args.num_gpus)
    else:
        # 多机多卡
        nproc_per_node = args.nproc_per_node or torch.cuda.device_count()
        success = run_multi_node(
            args.script, args.nnodes, args.node_rank, 
            args.master_addr, args.master_port, nproc_per_node
        )
    
    if success:
        print("训练完成!")
    else:
        print("训练失败!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)