import time
import torch

from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def demo():
    # 加载模型
    # AutoModelForCausalLM	因果语言模型（文本生成）	GPT-2、Llama
    # AutoModelForSeq2SeqLM	序列到序列模型（翻译、摘要）	T5、BART
    # AutoModelForMaskedLM	掩码语言模型（填空、特征提取）	BERT、RoBERTa
    # AutoModelForQuestionAnswering	问答任务	BERT-QA、RoBERTa-QA
    model = AutoModelForCausalLM.from_pretrained(
        "./modelscope/Qwen/Qwen2.5-1.5B-Instruct",  # 先手动将模型下载到本地
        torch_dtype='auto',  # 使用auto会根据硬件配置情况自行选择精度，如果不设置此参数，默认使用float32
        device_map="auto"  # 如果有GPU，可以自动加载到GPU
    )
    # 可以打印查看模型的网络结构
    # 例如qwen2 1.5B 由28 层 Qwen2DecoderLayer 构成，每个 Decoder 主要的核心是 self_attention 和 mlp
    print(model)

    # 增加Lora结构之后，打印模型结构查看变化
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    # trainable params: 9,232,384 || all params: 1,552,946,688 || trainable%: 0.5945
    model.print_trainable_parameters()
    # 下面通过自行计算参与训练的参数量，与上面的参数量对比是否一致
    total_trainable_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_trainable_params += param.numel()

    print(f"参与训练的参数数量: {total_trainable_params}")

    # Lora 之后在每一层(q_proj这些线性层)都增加了一个 lora_A 和 lora_B 结构来实现降维升维的作用，
    print(model)

    # 对话测试

    # todo tokenizer具体是什么？
    tokenizer = AutoTokenizer.from_pretrained("./modelscope/Qwen/Qwen2.5-1.5B-Instruct")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt = "5月至今上腹靠右隐痛，右背隐痛带酸，便秘，喜睡，时有腹痛，头痛，腰酸症状？"
    messages = [
        {"role": "system", "content": '你是一个医疗方面的专家，可以根据患者的问题进行解答。'},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    start = time.time()
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    end = time.time()
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 初始回复：您的描述表明您可能患有慢性胃炎或者胃溃疡等疾病。建议尽快就医并做进一步检查以明确诊断，并根据医生的指导进行治疗。同时注意饮食健康，避免辛辣、油腻食物，保持良好的生活习惯和心态。
    print(f"耗时：{end-start}s，{response}")

if __name__ == '__main__':
    demo()


# target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# 核心参数：指定需要插入 LoRA 适配器的目标模块名称。这些模块通常是模型中负责注意力计算或 FeedForward 层的关键组件：
# q_proj/k_proj/v_proj：注意力机制中的查询（Query）、键（Key）、值（Value）投影层；
# o_proj：注意力输出的投影层；
# gate_proj/up_proj/down_proj：LLaMA 等模型中 FeedForward 层的门控、上采样、下采样投影层。
# 只有这些模块会被添加 LoRA 参数，其他模块不做修改。
# inference_mode=False
# 表示当前处于 “训练模式”（False），而非推理模式（True）。训练时会启用梯度计算，推理时则会合并 LoRA 权重以加速计算。
# r=8
# LoRA 的秩（rank），控制低秩矩阵的维度。r越小，参数量越少（计算量也越小），但可能损失表达能力；r=8是常见的中等配置。
# lora_alpha=32
# LoRA 的缩放因子，用于调整低秩矩阵输出的权重（实际更新量为 (W + BA) * (alpha/r)，其中 BA 是低秩矩阵乘积）。alpha 通常大于 r，增强更新的影响。
# lora_dropout=0.1
# 应用于 LoRA 适配器的 dropout 概率（10%），用于防止过拟合，仅在训练时生效。