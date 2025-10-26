import time
import torch
import sys

from tqdm import tqdm


def train_model(model, train_loader, val_loader, optimizer, gradient_accumulation_steps,
                device, num_epochs, model_output_dir, writer):
    batch_step = 0
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
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
                batch_step += 1
            # 100条数据打印一次 loss
            if (index % 100 == 0 and index != 0) or index == len(train_loader) - 1:
                time2 = time.time()
                tqdm.write(
                    f"{index}, epoch: {epoch} -loss: {str(loss)} ; "
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
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)