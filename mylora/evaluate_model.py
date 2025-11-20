import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from qa_dataset import QADataset
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import jieba
from tqdm import tqdm
import os

class ModelEvaluator:
    def __init__(self, original_model_path, lora_model_path, test_data_path, tokenizer_path=None):
        """
        初始化评估器
        
        Args:
            original_model_path: 原始模型路径
            lora_model_path: LoRA微调后的模型路径
            test_data_path: 测试数据路径
            tokenizer_path: 分词器路径，如果为None则使用原始模型路径
        """
        self.original_model_path = original_model_path
        self.lora_model_path = lora_model_path
        self.test_data_path = test_data_path
        self.tokenizer_path = tokenizer_path or original_model_path
        
        # 设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载分词器
        print("加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        
        # 初始化Rouge评分器
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        # 词干提取是一种文本预处理技术，通过去除词语的后缀（如英文的 -ing、-ed、-s，中文的 “的”“了” 等），将词语还原为其 “词干”（核心词根）。
        # 中文词语没有像英文那样的时态、单复数等形态变化，词干提取对提升匹配度的作用不大，反而可能因过度处理（如误删有效词根）导致匹配错误。
        # BLEU平滑函数
        self.smoothing_function = SmoothingFunction().method1
        
    def load_original_model(self):
        """加载原始模型"""
        print("加载原始模型...")
        model = AutoModelForCausalLM.from_pretrained(
            self.original_model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_lora_model(self):
        """加载LoRA微调后的模型（已合并的完整模型）"""
        print("加载LoRA微调后的模型...")
        # 直接加载已合并的完整模型
        model = AutoModelForCausalLM.from_pretrained(
            self.lora_model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_test_data(self):
        """加载测试数据"""
        print("加载测试数据...")
        test_data = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    test_data.append({
                        'question': data['question'],
                        'answer': data['answer']
                    })
        print(f"测试数据加载完成，共 {len(test_data)} 条")
        return test_data
    
    def generate_response(self, model, question, max_new_tokens=512):
        """生成模型回答"""
        messages = [
            {"role": "system", "content": "你是一个医疗方面的专家，可以根据患者的问题进行解答。"},
            {"role": "user", "content": question}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device) # 本质是为批量处理文本，最好输入文本列表

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        # generated_ids中包含输入，这一步骤可以去除输入部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # zip(model_inputs.input_ids, generated_ids)：对齐批量数据
        # model_inputs.input_ids：模型的输入序列（批量格式），形状为 (batch_size, input_length)（例如 batch_size=1 时，是 [[输入token1, 输入token2, ...]]）。
        # generated_ids：模型的完整输出序列（批量格式），形状为 (batch_size, total_length)（total_length = input_length + 回答长度）。
        # zip(...)：将输入序列和完整输出序列按 “批量中的每个样本” 一一配对，得到 (input_ids, output_ids) 这样的元组（每个元组对应一个样本）。
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # skip_special_tokens=True：解码时自动忽略特殊符号（如 <s>、</s>、[PAD] 等），只保留实际文本内容。
        return response
    
    def calculate_bleu4(self, reference, candidate):
        """计算BLEU-4分数"""
        # 使用jieba进行中文分词
        reference_tokens = list(jieba.cut(reference))
        candidate_tokens = list(jieba.cut(candidate))
        
        # 计算BLEU-4
        bleu_score = sentence_bleu(
            [reference_tokens],  # 参考答案可能有多个
            candidate_tokens, 
            weights=(0.25, 0.25, 0.25, 0.25), # 通过比较候选文本与参考文本的 n-gram（连续 n 个词）重叠度
            smoothing_function=self.smoothing_function
            # smoothing_function（平滑函数）的作用是：
            # 对 “未出现的 n-gram” 进行微小的分数补偿，避免因短文本或偶然不匹配导致评分失真；
            # 常见的平滑函数（如 SmoothingFunction().method4）会通过公式调整概率分布，让短文本的评分更合理。
        )
        return bleu_score
    
    def calculate_rouge(self, reference, candidate):
        """计算Rouge分数 - 修复中文支持问题"""
        return self.calculate_rouge_scores_fixed(reference, candidate)
    
    def calculate_rouge_scores_fixed(self, reference, candidate):
        """
        手动实现ROUGE分数计算，解决rouge_score库对中文的问题
        """
        # 分词
        ref_tokens = list(jieba.cut(reference))
        cand_tokens = list(jieba.cut(candidate))
        
        # 计算Rouge-1
        rouge1 = self.calculate_rouge_n(ref_tokens, cand_tokens, 1)
        
        # 计算Rouge-2
        rouge2 = self.calculate_rouge_n(ref_tokens, cand_tokens, 2)
        
        # 计算Rouge-L
        rougeL = self.calculate_rouge_l(ref_tokens, cand_tokens)
        
        return {
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL
        }

    def calculate_rouge_n(self, ref_tokens, cand_tokens, n):
        """计算Rouge-n分数"""
        from collections import Counter
        
        if len(ref_tokens) < n or len(cand_tokens) < n:
            return 0.0
        
        # 生成n-gram
        ref_ngrams = self.get_ngrams(ref_tokens, n)
        cand_ngrams = self.get_ngrams(cand_tokens, n)
        
        if len(cand_ngrams) == 0:
            return 0.0
        
        # 计算重叠
        ref_counter = Counter(ref_ngrams)
        cand_counter = Counter(cand_ngrams)
        
        overlap = 0
        for ngram in cand_counter:
            if ngram in ref_counter:
                overlap += min(cand_counter[ngram], ref_counter[ngram])
        
        # 计算precision, recall, f1
        precision = overlap / len(cand_ngrams) if len(cand_ngrams) > 0 else 0
        recall = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def get_ngrams(self, tokens, n):
        """生成n-gram"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams

    def calculate_rouge_l(self, ref_tokens, cand_tokens):
        """计算Rouge-L分数（基于最长公共子序列）"""
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return 0.0
        
        # 计算最长公共子序列长度
        lcs_length = self.lcs(ref_tokens, cand_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        # 计算precision, recall, f1
        precision = lcs_length / len(cand_tokens)
        recall = lcs_length / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def lcs(self, X, Y):
        """计算最长公共子序列长度"""
        m, n = len(X), len(Y)
        
        # 创建DP表
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i-1] == Y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def evaluate_model(self, model, test_data, model_name):
        """评估单个模型"""
        print(f"\n开始评估 {model_name}...")
        
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        # 创建结果保存目录
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存详细结果的文件
        detailed_results_file = os.path.join(results_dir, f"{model_name}_detailed_results.json")
        detailed_results = []
        
        for i, item in enumerate(tqdm(test_data, desc=f"评估{model_name}")):
            question = item['question']
            reference_answer = item['answer']
            
            # 生成模型回答
            try:
                generated_answer = self.generate_response(model, question)
                
                # 计算BLEU-4
                bleu_score = self.calculate_bleu4(reference_answer, generated_answer)
                bleu_scores.append(bleu_score)
                
                # 计算Rouge分数
                rouge_scores = self.calculate_rouge(reference_answer, generated_answer)
                rouge1_scores.append(rouge_scores['rouge1'])
                rouge2_scores.append(rouge_scores['rouge2'])
                rougeL_scores.append(rouge_scores['rougeL'])
                
                # 保存详细结果
                detailed_result = {
                    'index': i,
                    'question': question,
                    'reference_answer': reference_answer,
                    'generated_answer': generated_answer,
                    'bleu4': bleu_score,
                    'rouge1': rouge_scores['rouge1'],
                    'rouge2': rouge_scores['rouge2'],
                    'rougeL': rouge_scores['rougeL']
                }
                detailed_results.append(detailed_result)
                
                # 每10个样本打印一次进度
                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1}/{len(test_data)} 个样本")
                    
            except Exception as e:
                print(f"处理第 {i} 个样本时出错: {e}")
                continue
        
        # 保存详细结果
        with open(detailed_results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        # 计算平均分数
        avg_bleu4 = np.mean(bleu_scores) if bleu_scores else 0
        avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0
        avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0
        avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0
        
        results = {
            'model_name': model_name,
            'total_samples': len(test_data),
            'processed_samples': len(bleu_scores),
            'BLEU-4': avg_bleu4,
            'Rouge-1': avg_rouge1,
            'Rouge-2': avg_rouge2,
            'Rouge-L': avg_rougeL
        }
        
        return results
    
    def run_evaluation(self):
        """运行完整的评估流程"""
        print("开始模型评估...")
        
        # 加载测试数据
        test_data = self.load_test_data()
        
        # 评估原始模型
        original_model = self.load_original_model()
        original_results = self.evaluate_model(original_model, test_data, "Original_Model")
        
        # 清理GPU内存
        del original_model
        torch.cuda.empty_cache()
        
        # 评估LoRA模型
        lora_model = self.load_lora_model()
        lora_results = self.evaluate_model(lora_model, test_data, "LoRA_Model")
        
        # 清理GPU内存
        del lora_model
        torch.cuda.empty_cache()
        
        # 打印结果对比
        self.print_comparison(original_results, lora_results)
        
        # 保存结果
        self.save_results(original_results, lora_results)
        
        return original_results, lora_results
    
    def print_comparison(self, original_results, lora_results):
        """打印评估结果对比"""
        print("\n" + "="*60)
        print("模型评估结果对比")
        print("="*60)
        
        print(f"{'指标':<15} {'原始模型':<15} {'LoRA模型':<15} {'提升':<15}")
        print("-" * 60)
        
        metrics = ['BLEU-4', 'Rouge-1', 'Rouge-2', 'Rouge-L']
        
        for metric in metrics:
            original_score = original_results[metric]
            lora_score = lora_results[metric]
            improvement = lora_score - original_score
            improvement_pct = (improvement / original_score * 100) if original_score > 0 else 0
            
            print(f"{metric:<15} {original_score:<15.4f} {lora_score:<15.4f} {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        print("="*60)
    
    def save_results(self, original_results, lora_results):
        """保存评估结果"""
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存汇总结果
        summary_results = {
            'original_model': original_results,
            'lora_model': lora_results,
            'comparison': {
                'BLEU-4_improvement': lora_results['BLEU-4'] - original_results['BLEU-4'],
                'Rouge-1_improvement': lora_results['Rouge-1'] - original_results['Rouge-1'],
                'Rouge-2_improvement': lora_results['Rouge-2'] - original_results['Rouge-2'],
                'Rouge-L_improvement': lora_results['Rouge-L'] - original_results['Rouge-L']
            }
        }
        
        summary_file = os.path.join(results_dir, "evaluation_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估结果已保存到: {results_dir}/")
        print(f"汇总结果: {summary_file}")


def main():
    # 配置路径
    original_model_path = "./modelscope/Qwen/Qwen2.5-1.5B-Instruct"
    lora_model_path = "./lora_output_bf16"  # 已合并的LoRA模型路径
    test_data_path = "./data/test_lite_plus.json"
    
    # 创建评估器
    evaluator = ModelEvaluator(
        original_model_path=original_model_path,
        lora_model_path=lora_model_path,
        test_data_path=test_data_path
    )
    
    # 运行评估
    try:
        original_results, lora_results = evaluator.run_evaluation()
        print("\n评估完成！")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()