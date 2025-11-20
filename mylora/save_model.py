import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "./modelscope/Qwen/Qwen2.5-1.5B-Instruct"
    lora_dir = "output_bf16"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_dir).to(device)
    print(model)
    # 合并model, 同时保存 token
    model = model.merge_and_unload()
    model.save_pretrained("lora_output_bf16")
    tokenizer.save_pretrained("lora_output_bf16")

if __name__ =='__main__':
    merge_model()