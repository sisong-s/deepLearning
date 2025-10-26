import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_lora():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 合并
    # model_path = "./modelscope/Qwen/Qwen2.5-1.5B-Instruct"
    # lora_dir = "output"

    
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', device_map='auto')
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = PeftModel.from_pretrained(model, lora_dir)
    # 直接用
    lora_model_path = './lora_output'
    model = AutoModelForCausalLM.from_pretrained(lora_model_path, torch_dtype='auto', device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

    model.to(device)
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
    model_inputs = tokenizer([text], return_tensors="pt").to(device) # 本质是为批量处理文本，最好输入文本列表
    start = time.time()
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    end = time.time()
    # generated_ids中包含输入，这一步骤可以去除输入部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 实际测试确实有思考过程
    print(f"耗时：{end - start}s，{response}")


# 2. tokenize=False
# 含义：是否直接对拼接后的文本进行分词（转换为 input_ids）。
# False（默认）：返回拼接后的原始字符串（方便查看格式是否正确）；
# True：直接返回分词后的结果（input_ids 等，可直接作为模型输入）。
# 3. add_generation_prompt=True
# 含义：是否在对话末尾添加 “生成提示”（告诉模型 “接下来该你回复了”）。
# True：拼接完成后，自动添加模型指定的 “助手回复前缀”（如 assistant: 或 <|assistant|>）；
# False：不添加，适用于已包含完整对话（如多轮对话的历史记录）的场景。
if __name__ == '__main__':
    test_lora()