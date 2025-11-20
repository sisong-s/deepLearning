import os
import time
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

from qa_dataset import QADataset


def setup_distributed():
    """初始化分布式训练环境"""
    # 检查是否在分布式环境中
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 单机多卡情况
        rank = 0
        world_size = torch.cuda.device_count()
        local_rank = 0
        
    # 设置当前进程使用的GPU
    torch.cuda.set_device(local_rank)
    
    # 初始化进程组
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_fsdp_model(model, auto_wrap_policy=None):
    """创建FSDP包装的模型"""
    # 确保所有参数都是相同的数据类型
    target_dtype = torch.bfloat16
    
    # 检查并转换所有参数到目标数据类型
    for name, param in model.named_parameters():
        if param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)
    
    # 检查并转换所有缓冲区到目标数据类型
    for name, buffer in model.named_buffers():
        if buffer.dtype != target_dtype and buffer.dtype.is_floating_point:
            buffer.data = buffer.data.to(target_dtype)
    
    # 混合精度配置 - 使用bfloat16可以提高性能
    mixed_precision_policy = MixedPrecision(
        param_dtype=target_dtype,  # 所有模型参数（包括LoRA参数）
        reduce_dtype=torch.float32, # 梯度累积和通信
        buffer_dtype=target_dtype, # 缓冲区存储
    )
    
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,  # 重要：支持PEFT
        sync_module_states=True,  # 确保模块状态同步
    )
    
    return fsdp_model


def save_fsdp_model(model, output_dir, rank):
    """保存FSDP模型"""
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    # 等待所有进程
    dist.barrier()
    
    # 配置状态字典类型
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()
        
        if rank == 0:
            # 只在rank 0保存模型
            model.module.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")


def main():
    # 设置分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    # 模型和训练参数
    model_name = "./modelscope/Qwen/Qwen2.5-1.5B-Instruct"
    train_json_path = "./data/train_lite.json"
    val_json_path = "./data/val_lite.json"
    max_source_length = 128
    max_target_length = 256
    epochs = 3
    batch_size = 4
    lr = 1e-4
    gradient_accumulation_steps = 16
    lora_rank = 8
    lora_alpha = 32
    model_output_dir = "output_fsdp_native"
    logs_dir = "logs_fsdp_native"
    
    # 只在主进程打印信息
    if rank == 0:
        print(f"Starting training with {world_size} GPUs")
        print(f"Rank: {rank}, Local Rank: {local_rank}")
    
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    
    # 设置PEFT配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    
    # 确保所有参数都是bfloat16类型（包括新添加的LoRA参数）
    target_dtype = torch.bfloat16
    for name, param in model.named_parameters():
        if param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)
    
    if rank == 0:
        print("Model parameters converted to bfloat16")
    
    # 不使用自动包装策略，让FSDP只在顶层包装
    # 这样可以避免lm_head层被单独包装导致的问题
    auto_wrap_policy = None
    
    # 使用FSDP包装模型
    model = create_fsdp_model(model, auto_wrap_policy)
    
    if rank == 0:
        print("Model wrapped with FSDP")
    
    # 准备数据
    if rank == 0:
        print("Loading training data...")
    
    train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    training_set = QADataset(train_json_path, tokenizer, max_source_length, max_target_length)
    training_loader = DataLoader(training_set, **train_params)
    
    if rank == 0:
        print("Loading validation data...")
    
    val_params = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}
    val_set = QADataset(val_json_path, tokenizer, max_source_length, max_target_length)
    val_loader = DataLoader(val_set, **val_params)
    
    # 优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    
    # 日志记录（只在主进程）
    writer = None
    if rank == 0:
        writer = SummaryWriter(logs_dir)
    
    # 开始训练
    if rank == 0:
        print("Starting training...")
        start_time = time.time()
    
    train_model(
        model=model,
        train_loader=training_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=epochs,
        model_output_dir=model_output_dir,
        writer=writer,
        rank=rank,
        world_size=world_size
    )
    
    if rank == 0:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"训练完成！总用时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    
    # 清理
    cleanup_distributed()


def train_model(model, train_loader, val_loader, optimizer, gradient_accumulation_steps,
                num_epochs, model_output_dir, writer, rank, world_size):
    batch_step = 0
    
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        
        # 创建进度条（只在主进程显示）
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Train Epoch: {epoch}")
        else:
            pbar = train_loader
        
        for index, data in enumerate(pbar):
            # 将数据移到GPU
            input_ids = data['input_ids'].cuda()
            attention_mask = data['attention_mask'].cuda()
            labels = data['labels'].cuda()
            
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # 梯度累积
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # 梯度更新
            if (index % gradient_accumulation_steps == 0 and index != 0) or index == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                
                # 记录日志（只在主进程）
                if rank == 0 and writer is not None:
                    writer.add_scalar('Loss/train', loss * gradient_accumulation_steps, batch_step)
                batch_step += 1
            
            # 打印损失（只在主进程）
            if rank == 0 and ((index % 100 == 0 and index != 0) or index == len(train_loader) - 1):
                time2 = time.time()
                tqdm.write(
                    f"{index}, epoch: {epoch} - loss: {loss:.6f} ; "
                    f"each step's time spent: {(time2 - time1) / (index + 0.0001):.4f}s"
                )
        
        # 验证
        model.eval()
        val_loss = validate_model(model, val_loader, rank, world_size)
        
        # 记录验证损失和保存模型（只在主进程）
        if rank == 0:
            if writer is not None:
                writer.add_scalar('Loss/val', val_loss, epoch)
            print(f'val_loss: {val_loss:.6f}, epoch: {epoch}')
            print(f'Saving model to {model_output_dir}')
        
        # 保存模型
        save_fsdp_model(model, model_output_dir, rank)


def validate_model(model, val_loader, rank, world_size):
    running_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(val_loader, desc="Validation")
        else:
            pbar = val_loader
            
        for data in pbar:
            input_ids = data['input_ids'].cuda()
            attention_mask = data['attention_mask'].cuda()
            labels = data['labels'].cuda()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # 收集所有进程的损失
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / world_size
            
            running_loss += loss.item()
            total_samples += 1
    
    return running_loss / total_samples if total_samples > 0 else 0.0


if __name__ == '__main__':
    main()