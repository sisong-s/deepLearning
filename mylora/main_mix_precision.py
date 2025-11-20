import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler


from qa_dataset import QADataset

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
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 使用bf16精度加载模型
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
    model.is_parallelizable = True
    model.model_parallel = True
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
    # 梯度缩放器，用于混合精度训练
    scaler = GradScaler()
    model = model.to(device)
    # 开始训练
    print("Start Training...")
    train_model(
        model=model,
        train_loader=training_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=scaler,
        gradient_accumulation_steps=gradient_accumulation_steps,
        device=device,
        num_epochs=epochs,
        model_output_dir=model_output_dir,
        writer=writer
    )

import time
import torch
import sys

from tqdm import tqdm


def train_model(model, train_loader, val_loader, optimizer, scaler, gradient_accumulation_steps,
                device, num_epochs, model_output_dir, writer):
    batch_step = 0
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        for index, data in enumerate(tqdm(train_loader, file=sys.stdout, desc="Train Epoch: " + str(epoch))):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            
            # 使用autocast进行bf16前向传播
            with autocast(dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps  # 对loss进行缩放
            
            # 使用scaler进行反向传播，梯度累积使用float32
            scaler.scale(loss).backward()
            
            # 梯度累积步数
            if (index % gradient_accumulation_steps == 0 and index != 0) or index == len(train_loader) - 1:
                # 更新网络参数
                scaler.step(optimizer)
                scaler.update()
                # 清空过往梯度
                optimizer.zero_grad()
                writer.add_scalar('Loss/train', loss * gradient_accumulation_steps, batch_step)
                batch_step += 1
            # 100条数据打印一次 loss
            if (index % 100 == 0 and index != 0) or index == len(train_loader) - 1:
                time2 = time.time()
                tqdm.write(
                    f"{index}, epoch: {epoch} -loss: {str(loss * gradient_accumulation_steps)} ; "
                    f"each step's time spent: {(str(float(time2 - time1) / float(index + 0.0001)))}")
        # 验证
        model.eval()
        val_loss = validate_model(model, val_loader, device)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f'val_loss: {val_loss}, epoch: {epoch}')
        print('Save Model To', model_output_dir)
        # 保存的模型只包含微调的参数部分，后面还需要合并模型
        model.save_pretrained(model_output_dir)


def validate_model(model, val_loader, device):
    running_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader, file=sys.stdout, desc="Validation Data")):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            
            # 验证时也使用bf16
            with autocast(dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)

if __name__ == '__main__':
    main()