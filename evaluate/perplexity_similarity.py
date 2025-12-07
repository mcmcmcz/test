import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import warnings
import os
import json
import argparse
warnings.filterwarnings("ignore")

def load_data(data_dir):
    folders = [folder for folder in os.listdir(data_dir) if 'pdf' in folder]
    all_data = {}
    for folder in folders:
        path = data_dir + folder + '/' + 'introduction_results.json'
        if os.path.exists(path): 
            with open(path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            if 'sections' in data.keys() and 'introduction' in data.keys():
                all_data[folder] = data
    return all_data

# /home/mczhang/zmc-dl/LLM/gpt2
# /home/mczhang/zmc-dl/LLM/NTP/SBERT
def calculate_gpt2_perplexity(text: str, model_name: str = "/home/mczhang/zmc-dl/LLM/gpt2", max_length: int = 2048) -> float:
    """
    使用GPT-2模型计算句子的困惑度
    
    Args:
        text (str): 要计算困惑度的句子
        model_name (str): GPT-2模型名称，默认为"gpt2"
        max_length (int): 最大序列长度，超过将被截断，默认为1024
    
    Returns:
        float: 句子的困惑度值
    """
    try:
        # 加载模型和分词器
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 获取模型的最大位置编码长度
        model_max_length = getattr(model.config, 'n_positions', 2048)
        effective_max_length = min(max_length, model_max_length)
        
        # 对文本进行分词，限制最大长度
        encodings = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=effective_max_length,
            truncation=True,
            padding=False
        )
        
        input_ids = encodings.input_ids.to(device)
        
        # 确保输入不为空
        if input_ids.size(1) == 0:
            print(f"警告: 输入文本为空或被完全截断")
            return float('inf')
        
        # 如果序列被截断，给出警告
        if input_ids.size(1) >= effective_max_length:
            print(f"警告: 文本长度超过{effective_max_length}个token，已被截断")
        
        # 计算困惑度
        with torch.no_grad():
            model.eval()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA错误，尝试使用CPU: {e}")
            # 强制使用CPU重试
            return _calculate_perplexity_cpu(text, model_name, max_length)
        else:
            raise e
    except Exception as e:
        print(f"计算困惑度时发生错误: {e}")
        return float('inf')


def _calculate_perplexity_cpu(text: str, model_name: str = "/home/mczhang/zmc-dl/LLM/gpt2", max_length: int = 2048) -> float:
    """
    强制使用CPU计算困惑度的备用函数
    """
    try:
        device = torch.device("cpu")
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_max_length = getattr(model.config, 'n_positions', 2048)
        effective_max_length = min(max_length, model_max_length)
        
        encodings = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=effective_max_length,
            truncation=True,
            padding=False
        )
        
        input_ids = encodings.input_ids.to(device)
        
        if input_ids.size(1) == 0:
            return float('inf')
        
        with torch.no_grad():
            model.eval()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()
        
    except Exception as e:
        print(f"CPU计算也失败: {e}")
        return float('inf')


def calculate_sentence_similarity(sentence1: str, sentence2: str, 
                                model_name: str = "/home/mczhang/zmc-dl/LLM/NTP/SBERT") -> float:
    """
    使用SentenceBERT计算两个句子的相似度
    
    Args:
        sentence1 (str): 第一个句子
        sentence2 (str): 第二个句子
        model_name (str): SentenceBERT模型名称，默认为"all-MiniLM-L6-v2"
    
    Returns:
        float: 两个句子的余弦相似度 (范围: -1 到 1)
    """
    # 加载SentenceBERT模型
    model = SentenceTransformer(model_name)
    
    # 编码句子
    embeddings = model.encode([sentence1, sentence2])
    
    # 计算余弦相似度
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    
    return float(similarity)


def batch_calculate_perplexity(texts: List[str], model_name: str = "/home/mczhang/zmc-dl/LLM/gpt2", max_length: int = 2048) -> List[float]:
    """
    批量计算多个句子的困惑度
    
    Args:
        texts (List[str]): 句子列表
        model_name (str): GPT-2模型名称
        max_length (int): 最大序列长度，默认为1024
    
    Returns:
        List[float]: 困惑度列表
    """
    perplexities = []
    for i, text in enumerate(texts):
        print(f"正在处理第 {i+1}/{len(texts)} 个文本...")
        ppl = calculate_gpt2_perplexity(text, model_name, max_length)
        perplexities.append(ppl)
    
    return perplexities


def batch_calculate_similarity(sentence_pairs: List[tuple], 
                             model_name: str = "/home/mczhang/zmc-dl/LLM/NTP/SBERT") -> List[float]:
    """
    批量计算多对句子的相似度
    
    Args:
        sentence_pairs (List[tuple]): 句子对列表，每个元素为(sentence1, sentence2)
        model_name (str): SentenceBERT模型名称
    
    Returns:
        List[float]: 相似度列表
    """
    similarities = []
    for sentence1, sentence2 in sentence_pairs:
        sim = calculate_sentence_similarity(sentence1, sentence2, model_name)
        similarities.append(sim)
    
    return similarities


def calculate_long_text_perplexity(text: str, model_name: str = "/home/mczhang/zmc-dl/LLM/gpt2", 
                                 chunk_size: int = 2048, overlap: int = 50) -> float:
    """
    计算长文本的困惑度，通过分块处理避免序列长度限制
    
    Args:
        text (str): 长文本
        model_name (str): GPT-2模型名称
        chunk_size (int): 每个块的大小（token数量）
        overlap (int): 块之间的重叠token数量
    
    Returns:
        float: 整个文本的平均困惑度
    """
    try:
        # 加载分词器
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # 对整个文本进行分词
        tokens = tokenizer.encode(text)
        
        if len(tokens) <= chunk_size:
            # 如果文本较短，直接计算
            return calculate_gpt2_perplexity(text, model_name, chunk_size)
        
        # 分块处理
        perplexities = []
        step = chunk_size - overlap
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size]
            if len(chunk_tokens) < 10:  # 跳过太短的块
                continue
            
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunk_ppl = calculate_gpt2_perplexity(chunk_text, model_name, chunk_size)
            
            if chunk_ppl != float('inf'):
                perplexities.append(chunk_ppl)
        
        if not perplexities:
            return float('inf')
        
        # 返回平均困惑度
        return sum(perplexities) / len(perplexities)
        
    except Exception as e:
        print(f"计算长文本困惑度时发生错误: {e}")
        return float('inf')



if __name__ == "__main__":
    # 示例使用
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_type", default="gpt_elaborate")
    args = parser.parse_args()
    print(args.generate_type)
    ft = load_data(f'../writing_agents_results_gpt/{args.generate_type}/')
    # with open('../results/naive_llama.json','r',encoding='utf-8') as fp:
    #     ft = json.load(fp)
    print(len(ft))
    # 测试困惑度计算
    print("=== GPT-2 困惑度计算示例 ===")
    # test_sentences = [
    #     "The quick brown fox jumps over the lazy dog.",
    #     "This is a well-formed English sentence.",
    #     "Random words: elephant purple mathematics flying."
    # ]
    ppl_results = {}
    for item in ft.keys():
        introduction = ft[item]['introduction']
        ppl = calculate_gpt2_perplexity(introduction)
        print(f"{item} ppl: {ppl}")
        ppl_results[item] = ppl

    with open(f'../writing_agents_results_gpt/{args.generate_type}_ppl.json','w',encoding='utf-8') as fw:
        json.dump(ppl_results, fw, ensure_ascii=False, indent=4)

    import numpy as np
    mean_ppl = np.mean([item[1] for item in ppl_results.items()])
    std_ppl = np.std([item[1] for item in ppl_results.items()])
    print(mean_ppl,std_ppl)
    
    # for i, sentence in enumerate(test_sentences, 1):
    #     ppl = calculate_gpt2_perplexity(sentence)
    #     print(f"句子 {i}: {sentence}")
    #     print(f"困惑度: {ppl:.2f}\n")
    
    # print("=== SentenceBERT 相似度计算示例 ===")
    # sentence_pairs = [
    #     ("I love programming", "I enjoy coding"),
    #     ("The weather is nice today", "It's a beautiful day"),
    #     ("Dogs are loyal animals", "Cats are independent pets")
    # ]
    
    # for i, (s1, s2) in enumerate(sentence_pairs, 1):
    #     similarity = calculate_sentence_similarity(s1, s2)
    #     print(f"句子对 {i}:")
    #     print(f"  句子1: {s1}")
    #     print(f"  句子2: {s2}")
    #     print(f"  相似度: {similarity:.4f}\n")
