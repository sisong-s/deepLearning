from accelerate import Accelerator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 【FSDP修改1】导入FSDP相关模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen2 import Qwen2DecoderLayer  # 【FSDP修改2】导入模型的transformer层


from qa_dataset import QADataset

def main():
    model_name = "./modelscope/Qwen/Qwen2.5-1.5B-Instruct"
    train_json_path = "./data/train_lite.json"
    val_json_path = "./data/val_lite.json"
    max_source_length = 128   # todo 输入长度最大可以设置为多少？
    max_target_length = 256   # todo 输出呢？
    epochs = 3
    batch_size = 4   # todo 显存大了之后可以增大，如何控制多卡训练
    lr = 1e-4
    gradient_accumulation_steps = 16
    lora_rank = 8   # 8或16或32
    lora_alpha = 32
    model_output_dir = "output_fsdp"  # 【FSDP修改3】修改输出目录名称以区分
    logs_dir = "logs_fsdp"  # 【FSDP修改4】修改日志目录名称以区分
    
    # 【FSDP修改5】初始化Accelerator时需要考虑FSDP配置
    # FSDP会自动从配置文件读取设置，但我们可以在这里做一些额外配置
    accelerator = Accelerator()
    
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 使用accelerate 混合精度训练bf16,这里也设置为bfloat16，否则可能会导致冲突报错
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
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
    
    # 【FSDP修改6】FSDP不需要手动设置并行化标志，FSDP会自动处理
    # model.is_parallelizable = True # 注释掉，FSDP自动处理
    # model.model_parallel = True # 注释掉，FSDP自动处理
    
    print("start load train data...")
    # shuffle=True：打乱训练数据的顺序, # num_workers=0：设置数据加载的进程数
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

    # 【FSDP修改7】使用accelerator.prepare准备模型和数据
    # FSDP的包装会在accelerator.prepare中自动完成
    model, optimizer, train_data, val_data = accelerator.prepare(model, optimizer, training_loader, val_loader)

    # 开始训练
    print("Start Training...")
    start_time = time.time()
    train_model(
        model=model,
        train_loader=train_data,
        val_loader=val_data,
        optimizer=optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=epochs,
        model_output_dir=model_output_dir,
        writer=writer,
        accelerator=accelerator
    )
    end_time = time.time()
    total_time = end_time - start_time
    print(f"训练完成！总用时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    accelerator.free_memory()

import time
import torch
import sys

from tqdm import tqdm


def train_model(model, train_loader, val_loader, optimizer, gradient_accumulation_steps,
                num_epochs, model_output_dir, writer, accelerator):
    batch_step = 0
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        for index, data in enumerate(tqdm(train_loader, file=sys.stdout, desc="Train Epoch: " + str(epoch))):
            # accelerate 会自动处理设备分配，不需要手动 .to(device)
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            labels = data['labels']
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss   # 交叉熵损失函数计算得来
            
            # 使用 accelerate 的梯度累积
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            
            # 梯度累积步数
            if (index % gradient_accumulation_steps == 0 and index != 0) or index == len(train_loader) - 1:
                # 更新网络参数
                optimizer.step()
                # 清空过往梯度
                optimizer.zero_grad()
                # 只在主进程记录日志
                if accelerator.is_main_process:
                    writer.add_scalar('Loss/train', loss * gradient_accumulation_steps, batch_step)
                batch_step += 1
            # 100条数据打印一次 loss
            if (index % 100 == 0 and index != 0) or index == len(train_loader) - 1:
                time2 = time.time()
                tqdm.write(
                    f"{index}, epoch: {epoch} -loss: {str(loss)} ; "
                    f"each step's time spent: {(str(float(time2 - time1) / float(index + 0.0001)))}")
        # 验证
        model.eval()
        val_loss = validate_model(model, val_loader, accelerator)
        # 只在主进程记录日志和保存模型
        if accelerator.is_main_process:
            writer.add_scalar('Loss/val', val_loss, epoch)
            print(f'val_loss: {val_loss}, epoch: {epoch}')
            print('Save Model To', model_output_dir)
            # 【FSDP修改8】FSDP模型保存方式
            # FSDP需要特殊的保存方式，accelerator.unwrap_model会处理FSDP的状态字典收集
            accelerator.unwrap_model(model).save_pretrained(model_output_dir)


def validate_model(model, val_loader, accelerator):
    running_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader, file=sys.stdout, desc="Validation Data")):
            # accelerate 会自动处理设备分配
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            labels = data['labels']
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # 【FSDP修改9】使用 accelerate 收集所有进程的损失
            # FSDP中这个操作会自动处理跨分片的损失聚合
            loss = accelerator.gather(loss).mean()
            running_loss += loss.item()
    return running_loss / len(val_loader)

if __name__ == '__main__':
    main()