import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from qa_dataset import QADataset
from memory_monitor import MemoryMonitor

def main():
    model_name = "./modelscope/Qwen/Qwen2.5-1.5B-Instruct"
    train_json_path = "./data/train_lite.json"
    val_json_path = "./data/val_lite.json"
    max_source_length = 128   #  输入长度可根据数据集调整，显存会随之变化
    max_target_length = 256   
    epochs = 5 # todo
    batch_size = 1   # 可根据显存使用情况调整，一般单卡很难设置的比较大
    lr = 1e-4
    gradient_accumulation_steps = 16
    lora_rank = 8   # 8或16或32
    lora_alpha = 32
    model_output_dir = "output"
    logs_dir = "logs"
    # 设备（这里先简单介绍单卡训练版本，后面会测试多卡训练）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # 初始化显存监控器
    memory_monitor = MemoryMonitor(device)
    memory_monitor.take_snapshot("start")
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    memory_monitor.take_snapshot("tokenizer_loaded")
    
    # 如果显存够，这里可以使用float32，不设置的话默认float32(1.5B模型8G显存使用float16、11G显存使用float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    
    # 在基础模型上启用梯度检查点（在应用LoRA之前）
    # if hasattr(model, 'gradient_checkpointing_enable'):
    #     model.gradient_checkpointing_enable()
    #     print("✓ 基础模型梯度检查点已启用")
    
    memory_monitor.take_snapshot("base_model_loaded")
    memory_monitor.print_memory_summary(model)
    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 任务类型：CAUSAL_LM 表示因果语言模型（Causal Language Model），即生成式任务
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "demo_proj"],
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True
    
    # 验证梯度检查点状态
    if hasattr(model.base_model.model, 'gradient_checkpointing') and model.base_model.model.gradient_checkpointing:
        print("✓ LoRA模型梯度检查点状态确认：已启用")
    else:
        print("⚠️ 梯度检查点可能未在基础模型上正确启用")
    
    memory_monitor.take_snapshot("lora_added")
    print("=== LoRA模型显存占用分析 ===")
    memory_monitor.print_memory_summary(model)
    print("start load train data...")
    train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    training_set = QADataset(train_json_path, tokenizer, max_source_length, max_target_length)
    training_loader = DataLoader(training_set, **train_params)
    print("start load validation data...")
    val_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    val_set = QADataset(val_json_path, tokenizer, max_source_length, max_target_length)
    val_loader = DataLoader(val_set, **val_params)
    # 日志记录
    writer = SummaryWriter(logs_dir)
    # 优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    model = model.to(device)
    
    # 额外的显存优化设置
    torch.backends.cudnn.benchmark = True  # 优化cudnn性能
    torch.cuda.empty_cache()  # 清理显存缓存
    
    # 打印可训练参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    memory_monitor.take_snapshot("model_to_device")
    print("=== 模型移动到GPU后显存占用 ===")
    memory_monitor.print_memory_summary(model)
    
    # 开始训练
    print("Start Training...")
    train_model(
        model=model,
        train_loader=training_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        device=device,
        num_epochs=epochs,
        model_output_dir=model_output_dir,
        writer=writer,
        memory_monitor=memory_monitor
    )

import time
import torch
import sys

from tqdm import tqdm


def train_model(model, train_loader, val_loader, optimizer, gradient_accumulation_steps,
                device, num_epochs, model_output_dir, writer, memory_monitor):
    batch_step = 0
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        
        # 每个epoch开始时记录显存状态
        memory_monitor.take_snapshot(f"epoch_{epoch}_start")
        if epoch == 0:  # 第一个epoch详细显示
            print(f"=== Epoch {epoch} 开始时显存状态 ===")
            memory_monitor.print_memory_summary(model, epoch=epoch)
        
        for index, data in enumerate(tqdm(train_loader, file=sys.stdout, desc="Train Epoch: " + str(epoch))):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss   # 交叉熵损失函数计算得来
            # 反向传播， 计算当前梯度
            loss.backward()
            # 梯度累积步数
            if (index % gradient_accumulation_steps == 0 and index != 0) or index == len(train_loader) - 1:
                # 更新网络参数
                optimizer.step()
                # 清空过往梯度
                optimizer.zero_grad()
                writer.add_scalar('Loss/train', loss, batch_step)
                
                # 记录显存信息到TensorBoard
                memory_monitor.log_memory_to_tensorboard(writer, batch_step, model)
                batch_step += 1
            # 100条数据打印一次 loss和显存信息
            if (index % 100 == 0 and index != 0) or index == len(train_loader) - 1:
                time2 = time.time()
                gpu_info = memory_monitor.get_gpu_memory_info()
                tqdm.write(
                    f"{index}, epoch: {epoch} -loss: {str(loss)} ; "
                    f"each step's time spent: {(str(float(time2 - time1) / float(index + 0.0001)))} ; "
                    f"GPU: {gpu_info['allocated']:.0f}MB/{gpu_info['total']:.0f}MB ({gpu_info['reserved']/gpu_info['total']*100:.1f}%)")
            # 释放显存
            del outputs, loss
            torch.cuda.empty_cache()  # 关键操作
        # 验证
        model.eval()
        val_loss = validate_model(model, val_loader, device)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f'val_loss: {val_loss}, epoch: {epoch}')
        
        # 每个epoch结束后显示详细显存信息
        print(f"=== Epoch {epoch} 结束时显存状态 ===")
        memory_monitor.print_memory_summary(model, epoch=epoch)
        
        print('Save Model To', model_output_dir)
        # 保存的模型只包含微调的参数部分，后面还需要合并模型
        model.save_pretrained(model_output_dir)
        
        # 清理显存缓存
        memory_monitor.clear_cache()


def validate_model(model, val_loader, device):
    running_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader, file=sys.stdout, desc="Validation Data")):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)

if __name__ == '__main__':
    main()